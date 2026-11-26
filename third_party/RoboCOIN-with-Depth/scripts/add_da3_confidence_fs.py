#!/usr/bin/env python3
"""仅为已有 LeRobot 数据集生成 DA3 深度置信度视频（不改已有 depth）。

输出:
  videos/chunk-*/observation.images.*_depth_conf*/episode_XXXXXX.mp4
  （gray12le，置信度按 episode percentile 映射到 0..4095）
  meta/confidence_norm.jsonl
  （每路相机每个 episode 的 min/max，用于还原原始 conf）

用法:
  python scripts/add_da3_confidence_fs.py \\
    --dataset-dir /path/to/dataset \\
    --da3-model depth-anything/DA3NESTED-GIANT-LARGE-1.1 \\
    --da3-process-res 504 --da3-chunk-size 420 --da3-overlap 140
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess as sp
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.depth_encoder import DepthVideoEncoder
from src.depth_repair.da3 import DA3DepthRepair, _make_chunks_streaming, _resolve_model_id


def _probe_size(path: Path) -> tuple[int, int] | None:
    probe = sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return None
    try:
        w, h = map(int, probe.stdout.strip().split(","))
    except ValueError:
        return None
    return w, h


def _decode_video(path: Path) -> list[np.ndarray]:
    size = _probe_size(path)
    if size is None:
        return []
    w, h = size
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = sp.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return []
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_bytes = w * h * 3
    n = len(raw) // frame_bytes
    return [raw[i * frame_bytes:(i + 1) * frame_bytes].reshape(h, w, 3) for i in range(n)]


class DA3ConfidenceExtractor(DA3DepthRepair):
    """复用 DA3 加载/分块逻辑，只返回置信度图（原分辨率 float32）。"""

    def extract_conf(self, rgb: list[np.ndarray]) -> list[np.ndarray]:
        if self._model is None:
            self._load_model()
        n = len(rgb)
        h, w = rgb[0].shape[:2]
        cs = self._chunk_size if self._chunk_size > 0 else n
        ol = self._overlap if cs < n else 0
        if ol <= 0 or cs >= n:
            return self._infer_conf_simple(rgb, cs, h, w)
        return self._infer_conf_streaming(rgb, n, cs, ol, h, w)

    def _infer_conf_chunk(
        self, batch_rgb: list[np.ndarray], h: int, w: int,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        imgs_cpu, _, _ = self._model._preprocess_inputs(
            batch_rgb, None, None, self._process_res, "upper_bound_resize")
        imgs, _, _ = self._model._prepare_model_inputs(imgs_cpu, None, None)
        raw = self._model.forward(imgs, None, None, export_feat_layers=[])
        pred = self._model._convert_to_prediction(raw)
        if pred.conf is None:
            raise RuntimeError("该 DA3 模型输出没有 conf / depth_conf")
        conf = torch.as_tensor(pred.conf, device=self._device).float()
        conf = F.interpolate(
            conf.unsqueeze(1), size=(h, w),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        out = conf.cpu().numpy().astype(np.float32)
        del imgs_cpu, imgs, raw, pred, conf
        torch.cuda.empty_cache()
        return out

    def _infer_conf_simple(
        self, rgb: list[np.ndarray], cs: int, h: int, w: int,
    ) -> list[np.ndarray]:
        import torch

        n = len(rgb)
        results: list[np.ndarray | None] = [None] * n
        start = 0
        while start < n:
            end = min(start + cs, n)
            try:
                batch = self._infer_conf_chunk(rgb[start:end], h, w)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cs <= 1:
                    raise
                cs = max(1, cs // 2)
                logger.warning(f"CUDA OOM, retry chunk_size={cs}")
                continue
            for j in range(len(batch)):
                results[start + j] = batch[j]
            start = end
        return results  # type: ignore[return-value]

    def _infer_conf_streaming(
        self, rgb: list[np.ndarray], n: int,
        cs: int, ol: int, h: int, w: int,
    ) -> list[np.ndarray]:
        import cv2

        chunks = _make_chunks_streaming(n, cs, ol)
        chunk_confs: list[np.ndarray] = []
        for start, end in chunks:
            chunk_confs.append(self._infer_conf_chunk(rgb[start:end], h, w))

        # 重叠区线性混合（与深度 streaming 对齐方式一致，但不做 scale/shift）
        for ci in range(1, len(chunks)):
            c_prev = chunk_confs[ci - 1][-ol:]
            weight = np.linspace(0, 1, ol, dtype=np.float32).reshape(ol, 1, 1)
            chunk_confs[ci][:ol] = (1 - weight) * c_prev + weight * chunk_confs[ci][:ol]

        results = [np.zeros((h, w), dtype=np.float32) for _ in range(n)]
        for ci, (start, end) in enumerate(chunks):
            local = chunk_confs[ci]
            c_local = len(local)
            ks, ke = 0, c_local if ci == len(chunks) - 1 else c_local - ol
            for j in range(ks, ke):
                # 已在模型分辨率插值到原图；这里直接写入
                results[start + j] = local[j]
                # 若形状意外不一致再 resize
                if local[j].shape != (h, w):
                    cv2.resize(local[j], (w, h), dst=results[start + j],
                               interpolation=cv2.INTER_LINEAR)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="为 LeRobot 数据集单独生成 DA3 置信度视频")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--da3-model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument("--da3-chunk-size", type=int, default=420)
    parser.add_argument("--da3-overlap", type=int, default=140)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=6)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--percentile", type=float, default=99.0,
                        help="置信度归一化分位数（映射到 0..4095）")
    parser.add_argument("--cameras", default="",
                        help="逗号分隔相机子串过滤，空=全部 rgb 目录")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    if args.log_file:
        logger.add(args.log_file, level="DEBUG")

    root = Path(args.dataset_dir)
    videos = root / "videos"
    if not videos.is_dir():
        raise FileNotFoundError(videos)

    rgb_dirs = sorted(videos.glob("chunk-*/observation.images.*rgb*"))
    if args.cameras.strip():
        keys = [k.strip() for k in args.cameras.split(",") if k.strip()]
        rgb_dirs = [d for d in rgb_dirs if any(k in d.name for k in keys)]
    if not rgb_dirs:
        raise FileNotFoundError(f"No RGB dirs under {videos}")

    extractor = DA3ConfidenceExtractor(
        model_id=args.da3_model,
        process_res=args.da3_process_res,
        chunk_size=args.da3_chunk_size,
        overlap=args.da3_overlap,
    )
    encoder = DepthVideoEncoder(fps=args.fps, crf=args.crf, preset=args.preset)
    norm_path = root / "meta" / "confidence_norm.jsonl"
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    # 追加写；同 key 重复时后写覆盖语义由读取方处理
    norm_records: list[dict] = []

    logger.info(f"model={_resolve_model_id(args.da3_model)}")
    logger.info(f"cameras={len(rgb_dirs)}  res={args.da3_process_res}  "
                f"chunk={args.da3_chunk_size}  overlap={args.da3_overlap}")

    for rgb_dir in rgb_dirs:
        conf_dir = rgb_dir.parent / rgb_dir.name.replace("rgb", "depth_conf")
        conf_dir.mkdir(parents=True, exist_ok=True)
        mp4s = sorted(rgb_dir.glob("episode_*.mp4"))
        pending = []
        for mp4 in mp4s:
            out = conf_dir / mp4.name
            tmp = conf_dir / (mp4.name + ".tmp")
            if out.is_file() and out.stat().st_size > 0:
                continue
            if tmp.is_file() and tmp.stat().st_size > 0:
                continue
            pending.append(mp4)
        logger.info(f"{rgb_dir.name}: {len(mp4s) - len(pending)}/{len(mp4s)} done, "
                    f"{len(pending)} remaining → {conf_dir.name}")
        if not pending:
            continue

        for mp4 in tqdm(pending, desc=conf_dir.name):
            frames = _decode_video(mp4)
            if not frames:
                logger.warning(f"decode failed: {mp4}")
                continue
            confs = extractor.extract_conf(frames)
            # per-episode percentile → gray12le
            stacked = np.stack(confs, axis=0)
            lo = float(np.percentile(stacked, 100.0 - args.percentile))
            hi = float(np.percentile(stacked, args.percentile))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(stacked)), float(np.nanmax(stacked))
            if hi <= lo:
                hi = lo + 1e-6
            out_mp4 = conf_dir / mp4.name
            encoder.encode_from_arrays(
                confs, str(out_mp4), normalize=(lo, hi))
            os.rename(str(out_mp4), str(out_mp4) + ".tmp")
            norm_records.append({
                "camera": conf_dir.name,
                "episode": mp4.name,
                "min": lo,
                "max": hi,
                "percentile": args.percentile,
                "n_frames": len(confs),
            })
            logger.info(
                f"[ok] {conf_dir.name}/{mp4.name}  "
                f"frames={len(confs)}  conf=[{lo:.4g},{hi:.4g}]")

        for fn in conf_dir.iterdir():
            if fn.name.endswith(".mp4.tmp"):
                os.rename(str(fn), str(fn)[:-4])

    if norm_records:
        with open(norm_path, "a") as f:
            for rec in norm_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"wrote {len(norm_records)} norms → {norm_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
