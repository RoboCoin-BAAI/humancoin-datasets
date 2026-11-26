"""Run calibrated left-eye hand reconstruction over an explicit L2 session list."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import traceback


def plan(sessions_file):
    tasks = []
    missing = []
    for session in dict.fromkeys(json.loads(sessions_file.read_text())):
        root = Path(session)
        episodes = sorted((root / "data").glob("chunk-*/episode_*"))
        if not episodes:
            missing.append(f"No episodes: {root}")
            continue
        for episode in episodes:
            chunk, name = episode.parent.name, episode.name
            output = root / "annotation" / chunk / "hand_pose_left_rectified" / name
            prefix = f"{chunk}_{name}_" if len(episodes) > 1 else ""
            manifest = root / "calibrations" / f"{prefix}dataset_manifest.json"
            if not manifest.is_file():
                missing.append(f"Missing manifest: {manifest}")
                continue
            k = json.loads(manifest.read_text())["intrinsics"]["left"]
            matrix = [[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]]
            video = root / "videos" / chunk / "observation.images.headband_left_rgb_rectified" / f"{name}.mp4"
            if not video.is_file():
                missing.append(f"Missing video: {video}")
                continue
            tasks.append((video, output, matrix, episode))
    return tasks, missing


def worker(gpu, tasks):
    os.environ["HAND_POSE_COMPILE_CACHE"] = str(
        Path(__file__).resolve().parent / ".cache" / "l2" / f"gpu-{gpu}"
    )
    from runtime.pipeline import Predictor
    from hand_points_l2 import export_hand_points

    predictor = None
    failed = 0
    for video, output, matrix, episode in tasks:
        try:
            if output.exists():
                if not all((output / name).is_file() and (output / name).stat().st_size
                           for name in ("prediction.npz", "metadata.json")):
                    raise ValueError(f"Incomplete existing output: {output}")
                print(f"GPU {gpu} reuse prediction {output}", flush=True)
            else:
                if predictor is None:
                    predictor = Predictor(device=f"cuda:{gpu}")
                print(f"GPU {gpu} start {video}", flush=True)
                predictor.run(video, output, geometry_intrinsics=matrix, render_overlay=False)
            counts = export_hand_points(output / "prediction.npz", episode, matrix)
            print(f"GPU {gpu} exported {episode}: left/right valid rows {counts}", flush=True)
        except Exception:
            failed += 1
            print(f"GPU {gpu} failed {video}", file=sys.stderr, flush=True)
            traceback.print_exc()
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-file", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus) or min(gpus) < 0:
        parser.error("GPU IDs must be unique nonnegative integers")
    tasks, missing = plan(args.sessions_file)
    for reason in missing:
        print(reason, file=sys.stderr, flush=True)
    print(f"{len(missing)} missing inputs", flush=True)
    if not tasks:
        parser.error("No usable episodes")
    print(f"{len(tasks)} episodes -> GPUs {gpus}", flush=True)
    if args.dry_run:
        return 0
    os.environ["PATH"] = os.pathsep.join(
        filter(None, (str(Path(sys.executable).parent), os.environ.get("PATH")))
    )
    import torch

    if max(gpus) >= torch.cuda.device_count():
        parser.error(f"Only {torch.cuda.device_count()} CUDA devices are visible")
    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(worker, gpu, tasks[i::len(gpus)]) for i, gpu in enumerate(gpus)]
        failed = sum(future.result() for future in futures)
    print(f"{len(tasks)} episodes, {failed} failed")
    return int(bool(failed or missing))


if __name__ == "__main__":
    raise SystemExit(main())
