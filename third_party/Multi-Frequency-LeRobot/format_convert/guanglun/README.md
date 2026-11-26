# Lightwheel EgoStandard MCAP conversion

`convert_guanglun_mcap.py` converts one MCAP into one episode in this repository's
multi-frequency v3.1 file layout. It does not download data or run QC, rectification,
VIO, hand inference or model annotation. Existing outputs are never overwritten.
The output is prepared in a sibling partial directory and renamed only on success.
Failed partial output is retained for inspection.

## Run locally

Dependencies: `numpy`, `pyarrow`, `pyyaml`, `mcap`, `protobuf`, `av`, plus an FFmpeg
executable. No torch, CUDA or model weights are needed for conversion. The shared
in-file protobuf decoder is reused from the Jianzhi converter.

PowerShell example for the downloaded workspace sample:

```powershell
$python = 'C:/Users/22563/Desktop/data/.venv-vio/Scripts/python.exe'
$ffmpeg = & $python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
& $python format_convert/guanglun/convert_guanglun_mcap.py `
  --source C:/Users/22563/Desktop/data/guanglun/032772f9-7e01-410e-b3c1-863bf4697017.mcap `
  --out C:/Users/22563/Desktop/data/guanglun/lerobot/new_sample `
  --ffmpeg $ffmpeg
```

Use a new output directory on each invocation. `lerobot/032772f9-...` already
contains the verified initial conversion.

## Feature mapping

| Source | Output |
| --- | --- |
| `/sensor/camera/head_left/video` | `observation.images.head_left_rectified` |
| `/sensor/camera/head_right/video` | `observation.images.head_right_rectified` |
| `/pose/head` | `observation.head_pose` |
| `/pose/headcam` | `observation.head_camera_pose` |
| `/pose/right_eye_cam` | `observation.right_eye_camera_pose` |
| `/pose/left_hand` | `observation.hand_left_points` + `observation.hand_left_quaternions` |
| `/pose/right_hand` | `observation.hand_right_points` + `observation.hand_right_quaternions` |
| `/pose/body`, `/pose/upper_body`, `/pose/lower_body` | `observation.body_pose`, `upper_body_pose`, `lower_body_pose` |
| `/annotation/bad_frame/...` | `observation.bad_frame_...` |

Head/body transforms remain `x,y,z,qx,qy,qz,qw`. Hand transforms are split into
63-dimensional world XYZ and 84-dimensional source `qx,qy,qz,qw` orientations.
Both follow the canonical 21-landmark order from the naming rules. The source
order was accepted after three-timepoint stereo projection visualization and
user confirmation; this is documented in each feature's `mapping_basis` rather
than represented as a provider-certified joint specification. No coordinate
transformation, quaternion normalization, interpolation or hand re-estimation is
performed. Source orientation conventions are preserved and their semantics are
not relabeled as MANO or MANUS parameters. No actions, IMU samples or depth maps
are fabricated from device declarations. The original MCAP remains the full source record; no `source_records` directory is exported.

## Time and video policy

- Header `ts` and protobuf camera timestamps are parsed as integer nanoseconds.
  Per-stream Parquet time is seconds relative to the first left-camera timestamp.
  The source receipt retains the integer time origin; per-stream offsets are stored as float64.
- Streams retain their own clocks; there is no nearest-neighbor resampling,
  forced stereo timestamp equality, interpolation, padding, or trimming.
- Left/right sample counts must match; duplicate or decreasing stream timestamps
  cause conversion to fail. Pose dimensionality must remain constant.
- H264/HEVC elementary streams are remuxed with `-c:v copy` into MP4. MP4 has the
  device's nominal FPS; native capture time remains in the timestamp Parquets.
  Source timing jitter is not reproduced in the CFR container.
- Conversion decodes the output once to verify frame count and the MP4 presentation
  timeline, without storing decoded images. B-frame inputs are conservatively
  rejected until explicit packet-to-display timestamp mapping is implemented.

## Pre-conversion pose jump gate

Before creating output directories or remuxing video, the converter checks only
head and both hand pose streams for severe adjacent-frame jumps. No pose repair,
interpolation, frame dropping or extra video QC is introduced by this gate.
Default limits at intervals of 1/30 second or less:

| Signal | Translation | Rotation |
| --- | --- | --- |
| Head | 0.20 m | 60 degrees |
| Any of 21 points/joints per hand | 0.30 m | 120 degrees |

For longer time intervals the allowed jump scales by `dt * 30`, so a normal
movement across a source gap is not judged as a one-frame jump. Rotation uses
the shortest quaternion angle, treating q and -q as equivalent. Missing/invalid
required head/hand streams or zero-length quaternions also block the conversion.
These are conservative engineering cutoffs, not validated accuracy guarantees.
Change thresholds with `--head-jump-m`, `--hand-jump-m`, `--head-jump-deg`, and
`--hand-jump-deg`. A failure exits with code 2 and prints the affected stream,
frame, joint index, measured jump and threshold. No output dataset is written.
Successful summaries are retained in `meta/guanglun_source.json/pose_jump_check`.

## Preserved metadata

The core metadata matches existing L1 conversions and the inherited metadata
writer from pinned `lerobot==0.3.3`:

```text
meta/
  info.json
  episodes.jsonl
  tasks.jsonl
  episodes_stats.jsonl
  guanglun_source.json
  guanglun_video_validation.json
```

`v3.1` in this repository's `info.json` describes the multi-frequency per-feature
data layout. It does not select upstream LeRobot v3 Parquet metadata. Do not mix
`tasks.parquet`, `meta/episodes/` or a standalone `stats.json` into this layout.
Per-episode numeric statistics are stored in `episodes_stats.jsonl` as
`{"episode_index": 0, "stats": {...}}`, matching L1. Video axis names are
`["h", "w", "c"]`; master-index timestamps are float32, while per-stream
timestamps retain float64 precision as in the repository's writer.

- `annotation/semantic_segments.json`: existing provider semantic annotations;
  not represented as freshly generated Argus results. Omitted protobuf scalar fields
  retain protobuf defaults (for example, an omitted segment start time is zero).
- `annotation/episode_000000_structured.json`: the provider segments mapped to
  the L1 structured annotation field layout. Only task descriptions, segment
  times/actions are mapped; each action label is the lowercase first word of its source description. Original skill labels remain in `semantic_segments.json`. Unsupported scalar fields
  are null and collections empty. Capture metadata stays in `meta/guanglun_source.json`, not in this annotation.
  This file maps existing provider text without inference and is not a complete
  Argus annotation or a successful Argus execution receipt. `timesteps_s` contains segment boundaries,
  not claims that video frames were sampled there. For already converted sessions,
  run `python format_convert/guanglun/format_semantics.py --session PATH`; existing
  structured annotations are never overwritten.
- `calibrations/source_camera_calibration.json`: compact fixed intrinsics and
  relative stereo transforms, after checking every source calibration row.
  Changing world camera poses remain in the head camera/right eye camera Parquets.
  Intrinsics changes, unmatched pose timestamps or a nonrigid stereo transform
  cause conversion to fail instead of silently collapsing varying calibration.
- `calibrations/dataset_manifest.json`: L1-style video paths, intrinsics, stereo
  transform, source-order mapping and synchronization information for the native
  rectified pair. Source translations are in metres, as confirmed by the user; `baseline_m`
  contains the computed stereo baseline (about 0.0642525 m for the sample).
  Native calibration retains small rotation/focal-length differences, so the
  simple disparity-to-depth formula uses the native rectified pair approximation.
- `calibrations/rectify_params.yaml`: native camera parameters and relative
  extrinsics. It deliberately omits fabricated R1/R2/P1/P2/Q matrices: no new
  rectification was applied. This is not an L1 rectification execution receipt.
- `meta/guanglun_source.json` / `collection_metadata`: full provider session metadata.
- `meta/guanglun_source.json`: source hash, time origin, per-topic counts, video
  validation and unresolved coordinate/rectification semantics.
- `meta/guanglun_video_validation.json`: video file identities, timestamp file
  hashes, and completed remux/frame-count validation results.
- `diagnostics/conversion_validation.json`: additional decoded-pixel and timestamp validation
  report produced for the downloaded real sample.

The file layout is checked against the repository's layout helpers. The conversion
and video/Parquet outputs have been tested locally; loading this new source through
the full torch/LeRobot training stack has not yet been tested. Existing L1/L3 QC
and rectification scripts assume their own calibration and IMU conventions and
must not be run on this source without an adapter.

## Hand point inspection

Run `python format_convert/guanglun/inspect_hand_points.py --session PATH` to
generate `diagnostics/hand_points/`: indexed 3D PNG, offline rotatable 3D HTML,
left/right video projection overlays at 5/15/25 seconds, and a JSON report.
Visualization additionally needs OpenCV, matplotlib and plotly. The overlay
assumes XYZ in world coordinates and inverts the native camera-to-world pose.
Candidate canonical finger chains are drawn explicitly as hypotheses. The sample
overlays visually support their consistency but do not certify exact anatomical
joint definitions or measurement accuracy. No dataset fields are changed.

Native camera videos are published with the `_rectified` suffix per user-confirmed source rectification and the sample epipolar check. This is a naming/provenance mapping only: the converter still uses stream copy and performs no rectification. Source frame IDs remain unchanged in the calibration records. Calibration entries use `head_left_rectified`/`head_right_rectified`, with explicit raw_camera_id and feature_key fields.
