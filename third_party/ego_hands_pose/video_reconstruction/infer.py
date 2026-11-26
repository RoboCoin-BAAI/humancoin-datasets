"""Reconstruct camera-space motion from one video."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp8"))
    parser.add_argument(
        "--resize-mode", choices=("center_crop", "long_edge_pad"), default="center_crop"
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overlay", choices=("anchors", "joints", "mesh"), default="joints")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--camera-json", type=Path, help="Original-video pixel intrinsics; visualization only"
    )
    parser.add_argument(
        "--geometry-camera-json",
        type=Path,
        help="Fixed original-video pixel intrinsics for decoder and 3D inference",
    )
    parser.add_argument("--global-consistency", action="store_true")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--window-batch-size", type=int)
    parser.add_argument("--rerender", type=Path, help="Existing raw result directory")
    args = parser.parse_args()
    if args.rerender and args.geometry_camera_json:
        parser.error(
            "--geometry-camera-json requires new inference; cannot be used with --rerender"
        )
    # Running a virtualenv's Python by absolute path does not activate its bin directory.
    env_bin = Path(sys.executable).parent
    current_path = os.environ.get("PATH")
    os.environ["PATH"] = os.pathsep.join(filter(None, (str(env_bin), current_path)))

    from runtime.pipeline import Predictor
    from runtime.runtime import ROOT

    predictor = Predictor(args.config, args.device, precision=args.precision)
    if args.no_compile:
        predictor.config["compile"] = False
    if args.window_batch_size is not None:
        if args.window_batch_size < 1:
            parser.error("window batch size must be positive")
        predictor.config["window_batch_size"] = args.window_batch_size
    k = json.loads(args.camera_json.read_text()) if args.camera_json else None
    geometry_k = (
        json.loads(args.geometry_camera_json.read_text()) if args.geometry_camera_json else None
    )
    common = dict(
        overlay_mode=args.overlay,
        threshold=args.threshold,
        intrinsics=k,
        global_correction=args.global_consistency,
    )
    if args.rerender:
        result = predictor.rerender(args.video, args.rerender, **common)
    else:
        output = args.output or ROOT / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        result = predictor.run(
            args.video,
            output,
            resize_mode=args.resize_mode,
            max_frames=args.max_frames,
            render_overlay=not args.no_overlay,
            geometry_intrinsics=geometry_k,
            **common,
        )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "summary"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
