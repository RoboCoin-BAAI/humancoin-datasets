# Dataset Field Naming Rules

## Conversion Scope

Conversion to HumanCoin format is the first processing step for externally
sourced datasets. Internally collected datasets keep their original collection
format and skip this conversion step.

The current planning estimate covers data from approximately 22 external
providers. Budget for one conversion script per provider as an initial estimate.
Confirm the source format and camera and sensor field mappings before deciding
which providers can share an implementation.


## General Rules

- Image features use `observation.images.<camera_name>`.
- Motion and sensor features use `observation.<name>`.
- Control targets use `action`.
- Converter-specific raw identifiers do not appear in final feature keys.
- Calibration and annotation metadata should refer to the same canonical names as
  the dataset features.

## Conversion Recommendations

### 1. Keep only the undistorted stereo pair by default

When converting source data to HumanCoin format, the recommended video output
contains two views: the undistorted left and right head cameras. Use
`observation.images.head_left` and `observation.images.head_right` for those
views. This default applies especially to L1 recordings captured in natural
settings, where additional cameras can capture identifiable faces.

The current downstream workflow does not use the outer side views. Including
their videos would add face detection and redaction work, with an extra round
of video decoding and encoding that increases processing time and compute
cost. Keeping the stereo pair still requires checking those two views for
identifiable faces when preparing data for release.

Keep the original side-view videos in the source archive. In the converted
dataset, retain side-camera information in metadata only, such as raw camera
IDs, calibration, and references to the original recordings where available.
The camera names below define how to name views when a workflow explicitly
includes them.

## Head Ring Cameras

Head-ring output names are canonical camera positions, not raw source IDs.
Converters must map each source's raw identifiers to the appropriate canonical
name in that converter or its config.

Camera image and optical-frame axes follow the OpenCV optical convention:

```text
image u: right
image v: down

camera optical frame:
  +x: right
  +y: down
  +z: forward along the optical axis
```

Allowed six-position head-ring names:

- `observation.images.head_left_outer1`
- `observation.images.head_left_outer0`
- `observation.images.head_left`
- `observation.images.head_right`
- `observation.images.head_right_outer0`
- `observation.images.head_right_outer1`

Do not publish final features as `camera0`, `camera1`, etc. Those raw IDs may be
stored in calibration metadata as provenance fields such as `raw_camera_id`.

## Wrist Cameras

Use:

- `observation.images.wrist_left`
- `observation.images.wrist_right`

If a wrist has multiple cameras, extend under the same side prefix, for example
`observation.images.wrist_left_inner` and `observation.images.wrist_left_outer`.

## Hand Features

Use MANO and 21-point features separately:

- `observation.hand_left_mano`: MANO parameters, 157 dimensions.
- `observation.hand_right_mano`: MANO parameters, 157 dimensions.
- `observation.hand_left_points`: 21 global xyz points flattened to 63 dimensions.
- `observation.hand_right_points`: 21 global xyz points flattened to 63 dimensions.
- `observation.hand_left_quaternions`: optional 21 source joint orientations,
  flattened to 84 dimensions in `qx, qy, qz, qw` order.
- `observation.hand_right_quaternions`: same orientation representation for the right hand.
- `observation.hand_left_joints`: MANUS-style joint angles, 20 dimensions.
- `observation.hand_right_joints`: MANUS-style joint angles, 20 dimensions.

Hand point features use this 21-landmark order:

```text
wrist,
thumbCMC, thumbMCP, thumbIP, thumbTip,
indexMCP, indexPIP, indexDIP, indexTip,
middleMCP, middlePIP, middleDIP, middleTip,
ringMCP, ringPIP, ringDIP, ringTip,
pinkyMCP, pinkyPIP, pinkyDIP, pinkyTip
```

Flatten point coordinates in point-major xyz order:

```text
wrist_x, wrist_y, wrist_z, thumbCMC_x, thumbCMC_y, thumbCMC_z, ...
```

Quaternion features follow the same landmark order, with names such as
`wrist_qx`, `wrist_qy`, `wrist_qz`, `wrist_qw`, `thumbCMC_qx`, etc.
Store the source rotation convention in feature metadata; do not reinterpret
these as MANO parameters or MANUS joint angles. Position features must record
their reference frame in metadata.

MANUS-style joint features are split by hand. Because the feature key already
contains `left` or `right`, per-dimension names do not include a side prefix:

```text
ThumbMCPSpread, ThumbMCPStretch, ThumbPIPStretch, ThumbDIPStretch,
IndexSpread, IndexMCPStretch, IndexPIPStretch, IndexDIPStretch,
MiddleSpread, MiddleMCPStretch, MiddlePIPStretch, MiddleDIPStretch,
RingSpread, RingMCPStretch, RingPIPStretch, RingDIPStretch,
PinkySpread, PinkyMCPStretch, PinkyPIPStretch, PinkyDIPStretch
```

## Motion Features

Use:

- `observation.imu`: IMU `[ax, ay, az, gx, gy, gz]`.
- `observation.head_pose`: head-mounted device pose in the dataset's main world frame.
- `observation.relative_head_pose`: relative head-mounted device pose when present.
- `observation.state`: generic robot state when the source is not a head-mounted device.
- `action`: action or target command.

When a source has multiple IMUs or robot states, add explicit position or device
qualifiers after the base name, such as `observation.imu_head` or
`observation.state_left_arm`.

### Derived VIO Head Pose

For a per-episode VIO trajectory without alignment to the dataset's main world
frame, use `observation.relative_head_pose`; reserve `observation.head_pose`
for poses actually aligned to that main world frame.

The L1 OpenVINS adapter stores `[x, y, z, qx, qy, qz, qw]`, with translation in
meters and a Hamilton xyzw quaternion. Its transform is
`T_first_valid_imu_current_imu`, using the head-mounted IMU device frame, with
an independent origin per episode. This does not imply an anatomical head
frame. Store the reference frame and source transform in provenance metadata.

Use `observation.relative_head_pose_status` for the accompanying numeric status
feature; its code mapping belongs in feature metadata. Preserve camera-aligned
sensor timestamps and original frame rows. Unavailable poses contain NaN and
must be filtered using status. Do not interpolate missing poses or assume a
basic quality check certifies trajectory accuracy.
