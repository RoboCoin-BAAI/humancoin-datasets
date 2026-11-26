"""
depth_augmenter_fs.py — 基于文件系统的深度增强器

直接读取 LeRobot 数据集文件（不依赖 lerobot Python API），
从 RGB 视频解码 → repair → 编码深度视频 → 更新 meta。
"""

from __future__ import annotations

import json
import os
import subprocess as sp
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.depth_encoder import DepthVideoEncoder


class DepthAugmenterFS:
    """文件系统级深度增强器。

    绕过 LeRobotDataset，直接操作 videos/ chunk 目录中的 mp4 文件。
    """

    def __init__(
        self,
        dataset_dir: str,
        strategy: str = "da3",
        fps: int = 30, crf: int = 6, preset: str = "slow",
        output_dir: str | None = None,
        cameras: list[str] | None = None,
        skip_rectified: bool = False,
        **repair_kwargs,
    ):
        self.root = Path(dataset_dir)
        # 深度与 meta 写入 out_root；默认写回原数据集
        self.out_root = Path(output_dir) if output_dir else self.root
        self.cameras = [c.strip() for c in (cameras or []) if c.strip()]
        self.skip_rectified = skip_rectified
        self._fps = fps
        self._crf = crf
        self._preset = preset

        from src.depth_repair import create_depth_repair
        self._repair = create_depth_repair(strategy, **repair_kwargs)
        self._encoder = DepthVideoEncoder(fps=fps, crf=crf, preset=preset)

    def _filter_rgb_dirs(self, rgb_dirs: list[Path]) -> list[Path]:
        out = []
        for d in rgb_dirs:
            if self.skip_rectified and "rectified" in d.name:
                continue
            if self.cameras and not _camera_match(d.name, self.cameras):
                continue
            out.append(d)
        return out

    def _depth_dir_for(self, rgb_dir: Path) -> Path:
        """RGB 目录 → 输出侧 depth 目录（保持 chunk-*/ 相对结构）。"""
        videos_src = self.root / "videos"
        rel = rgb_dir.relative_to(videos_src)  # chunk-000/observation.images.xxx_rgb
        depth_name = rgb_dir.name.replace("rgb", "depth")
        return self.out_root / "videos" / rel.parent / depth_name

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self, normalize: bool = False, log_file: str | None = None):
        """扫描所有 RGB 视频，生成深度视频并更新 meta（支持断点续跑）。"""
        # ---- 配置 loguru ----
        logger.remove()
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<level>{message}</level>"
            ),
            level="INFO",
            colorize=True,
        )
        if log_file:
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
                level="DEBUG",
                rotation="100 MB",
                retention="7 days",
            )
            logger.info(f"Log file: {log_file}")

        videos_dir = self.root / "videos"
        if not videos_dir.is_dir():
            raise FileNotFoundError(str(videos_dir))

        # 发现 RGB 相机目录
        rgb_dirs = sorted(videos_dir.glob("chunk-*/observation.images.*rgb*"))
        rgb_dirs = self._filter_rgb_dirs(rgb_dirs)
        if not rgb_dirs:
            raise FileNotFoundError(f"No matching RGB video dirs under {videos_dir}")

        src_info = self.root / "meta" / "info.json"
        src_stats = self.root / "meta" / "episodes_stats.jsonl"
        out_meta = self.out_root / "meta"
        out_meta.mkdir(parents=True, exist_ok=True)
        info_path = out_meta / "info.json"
        stats_path = out_meta / "episodes_stats.jsonl"

        with open(src_info) as f:
            info = json.load(f)
        # 独立输出时：meta 写到 out_root，不改源目录；首次从源复制 stats
        if self.out_root != self.root and not stats_path.exists() and src_stats.exists():
            stats_path.write_text(src_stats.read_text())
        elif self.out_root == self.root:
            stats_path = src_stats
            info_path = src_info

        video_params = self._get_video_params(info.get("features", {}))

        # ---- 断点续跑 ----
        completed = self._scan_progress(rgb_dirs)

        for rgb_dir in rgb_dirs:
            depth_dir = self._depth_dir_for(rgb_dir)
            depth_dir.mkdir(parents=True, exist_ok=True)
            depth_key = depth_dir.name
            done_set = completed.get(depth_key, set())

            mp4_files = sorted(rgb_dir.glob("episode_*.mp4"))
            total = len(mp4_files)
            size = _probe_size(mp4_files[0]) if mp4_files else None

            # 过滤已完成 episode
            pending = [p for p in mp4_files
                       if int(p.stem.split("_")[-1]) not in done_set]
            n_done = total - len(pending)

            if n_done > 0:
                logger.info(
                    f"{rgb_dir.name}: {n_done}/{total} already done, "
                    f"{len(pending)} remaining"
                )
            else:
                logger.info(f"{rgb_dir.name}: {total} episodes")

            # 确保 info 有 depth key（即使全部完成也要写入）
            self._inject_depth_to_info(info, depth_key, video_params, size)

            if not pending:
                continue

            all_ep_stats: list[dict] = []
            for mp4_path in tqdm(pending, desc="episodes"):
                # 解码 RGB 视频
                rgb_frames = _decode_video(mp4_path)
                if not rgb_frames:
                    continue

                # 推理深度
                depth_frames = self._repair.repair_frames(rgb_frames, [])

                # 时序平滑
                depth_frames = self._smooth_depth(depth_frames)

                # 编码深度视频
                depth_path = depth_dir / mp4_path.name
                all_ep_stats.append(self._encoder.compute_stats(depth_frames))

                if normalize:
                    dmin, dmax = self._encoder.normalize_params(depth_frames, percentile=95)
                    norm = (dmin, dmax)
                else:
                    norm = None
                    depth_frames = [np.clip(f, 0, 4095) for f in depth_frames]

                self._encoder.encode_from_arrays(depth_frames, str(depth_path), normalize=norm)
                os.rename(str(depth_path), str(depth_path) + ".tmp")

            # 注入 meta（只注入本次新处理的）
            self._inject_depth_to_stats(
                stats_path, depth_key, all_ep_stats, pending,
            )
            self._rename_tmp_files(depth_dir)

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        logger.info("Done.")

    # ------------------------------------------------------------------
    # 断点续跑
    # ------------------------------------------------------------------

    def _scan_progress(
        self,
        rgb_dirs: list[Path],
    ) -> dict[str, set[int]]:
        """返回每个 depth camera 已完成的 episode index 集合。

        规则：
        1. 每个 RGB 都有 depth .mp4 → chunk 完整跑完，全部跳过
        2. 否则：.tmp = 已完成（保留），.mp4 = 未完成（删除，重来）
        """
        completed: dict[str, set[int]] = {}

        for rgb_dir in rgb_dirs:
            depth_dir = self._depth_dir_for(rgb_dir)
            depth_key = depth_dir.name
            mp4_files = sorted(rgb_dir.glob("episode_*.mp4"))
            done: set[int] = set()

            # 规则 1：检查是否全部完成
            all_done = bool(mp4_files) and all(
                (depth_dir / f.name).exists() for f in mp4_files
            )

            if all_done:
                for mp4_path in mp4_files:
                    done.add(int(mp4_path.stem.split("_")[-1]))
            else:
                # 规则 2：.tmp = 已完成保留，.mp4 = 未完成删除
                for mp4_path in mp4_files:
                    ep_idx = int(mp4_path.stem.split("_")[-1])
                    fname = mp4_path.name
                    depth_mp4 = depth_dir / fname
                    depth_tmp = depth_dir / (fname + ".tmp")

                    if depth_tmp.exists():
                        done.add(ep_idx)
                        if depth_mp4.exists():
                            depth_mp4.unlink()
                    elif depth_mp4.exists():
                        depth_mp4.unlink()
                        logger.warning(
                            f"[resume] removed incomplete: "
                            f"{depth_dir.name}/{fname}"
                        )

            completed[depth_key] = done
            if done:
                logger.info(
                    f"[resume] {depth_key}: {len(done)}/"
                    f"{len(mp4_files)} done"
                )
            if not all_done and not done:
                logger.info(
                    f"[resume] {depth_key}: 0/{len(mp4_files)} done, "
                    f"starting fresh"
                )

        return completed

    # ------------------------------------------------------------------
    # meta helpers
    # ------------------------------------------------------------------

    def _smooth_depth(
        self, depth_frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """时序平滑。委托给 repair 对象。"""
        return self._repair.smooth_depth(depth_frames)

    @staticmethod
    def _get_video_params(features: dict) -> dict:
        for feat in features.values():
            if isinstance(feat, dict) and "info" in feat:
                return {
                    "video.fps": feat["info"].get("video.fps", 30),
                    "video.codec": "libx265",
                    "video.pix_fmt": "gray12le",
                }
        return {"video.fps": 30, "video.codec": "libx265", "video.pix_fmt": "gray12le"}

    @staticmethod
    def _inject_depth_to_info(
        info: dict, depth_key: str, video_params: dict,
        size: tuple[int, int] | None = None,
    ):
        """size 为 RGB 视频的 (W, H)；深度视频与之同分辨率。"""
        W, H = size if size else (640, 480)
        info.setdefault("features", {})[depth_key] = {
            "dtype": "video",
            "shape": [H, W],
            "names": ["height", "width"],
            "info": {**video_params, "video.height": H, "video.width": W},
        }

    @staticmethod
    def _inject_depth_to_stats(stats_path, depth_key, depth_stats, mp4_files):
        # 每路相机都会写一次，必须基于文件当前内容，否则会覆盖前面相机的统计
        with open(stats_path) as f:
            stats = [json.loads(line) for line in f if line.strip()]
        # 将 episode 文件名映射到 stats 条目（按 episode index）
        ep_indices = [int(p.stem.split("_")[-1]) for p in mp4_files]
        for ep_idx, ds in zip(ep_indices, depth_stats):
            if ep_idx < len(stats):
                stats[ep_idx].setdefault("stats", {})[depth_key] = ds
        with open(stats_path, "w") as f:
            for s in stats:
                f.write(json.dumps(s) + "\n")

    @staticmethod
    def _rename_tmp_files(video_dir: Path):
        for fn in video_dir.iterdir():
            if fn.name.endswith(".mp4.tmp"):
                os.rename(str(fn), str(fn)[:-4])


# ============================================================================
# helpers
# ============================================================================

def _camera_match(dir_name: str, cameras: list[str]) -> bool:
    """observation.images.<cam>_... 是否命中 cameras 前缀列表（如 head / eye）。"""
    if "observation.images." not in dir_name:
        return False
    rest = dir_name.split("observation.images.", 1)[1]
    for cam in cameras:
        if rest == cam or rest.startswith(cam + "_") or rest.startswith(cam + "."):
            return True
    return False


def _probe_size(path: Path) -> tuple[int, int] | None:
    """返回视频 (W, H)，失败返回 None。"""
    probe = sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        logger.warning(f"ffprobe failed for {path.name}")
        return None
    try:
        W, H = map(int, probe.stdout.strip().split(","))
    except ValueError:
        return None
    return W, H


def _decode_video(path: Path) -> list[np.ndarray]:
    """解码 H.265 RGB 视频为 uint8 (H, W, 3) 帧列表。"""
    size = _probe_size(path)
    if size is None:
        return []
    W, H = size

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-",
    ]
    proc = sp.run(cmd, capture_output=True)
    if proc.returncode != 0:
        logger.warning(f"decode failed for {path.name}")
        return []

    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_bytes = W * H * 3
    n_frames = len(raw) // frame_bytes
    frames = [raw[i * frame_bytes:(i + 1) * frame_bytes].reshape(H, W, 3)
              for i in range(n_frames)]
    return frames


