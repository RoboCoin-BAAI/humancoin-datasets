# Archebase 转换

将单条 `data.mcap` 和同目录的 `annotation.json` 转为仓库的多频 LeRobot v3.1 布局。不依赖 ROS 安装、torch 或 GPU。

在仓库根目录运行：

```powershell
python -m pip install -r format_convert/archebase/requirements.txt
python format_convert/archebase/convert_archebase_mcap.py `
  --source D:/data/archebase/186460 `
  --out D:/data/archebase/186460_lerobot
```

`--source` 可以指向单条数据目录或 `data.mcap`。每次输出一个 episode；已有输出拒绝覆盖。失败时保留 `.输出名.partial-<ID>` 供排查，成功后才发布到输出目录。`--task` 可指定任务名称，否则采用标注 `annoDetail.desc`，为空时采用目录名。

视频使用 FFmpeg 无损转封装（`-c:v copy`），PyAV 仅读取容器与编码信息，不解码视频。完整解码检查留给后续质检。元数据记录 `source_packets` 和 `decode_validation_performed=false`，不将源消息数当作已验证的解码帧数。FFmpeg 优先使用 PATH，可通过 `--ffmpeg` 指定，也可自动使用 imageio-ffmpeg。`--fps` 默认为 30，仅指定 MP4 的恒定播放帧率；源消息没有名义帧率字段，准确采集时间另存每路相机的 Parquet。

| 源数据 | 输出特征 |
| --- | --- |
| `/decxin/{left,right}_rgb/h264` | `observation.images.head_{left,right}` |
| `/decxin/imu` | `observation.imu`（ax/ay/az/gx/gy/gz） |
| `/ab/slam/pose` | `observation.head_pose`（xyz + xyzw，保留源 SLAM 位姿） |
| `/ab/pose/{Left,Right}_Hand` | `observation.hand_{left,right}_points`（63 维）、`*_quaternions`（84 维） |
| `/ab/pose/{Left,Right}_Hand_2D` | `*_points_2d`（42 维）、confidence |

保留各流原始采样率，不插值、不裁剪。时间以首张左相机图像为零点，先用整数纳秒相减再转 float64 秒；较早的 IMU 和标注允许负时间。每个特征独立存储 Parquet，左右视频各自保留时间戳。左相机定义 `master_index.parquet`。

手部按用户可视化确认的源顺序保存：wrist、thumbCMC/MCP/IP/Tip、indexMCP/PIP/DIP/Tip、middleMCP/PIP/DIP/Tip、ringMCP/PIP/DIP/Tip、pinkyMCP/PIP/DIP/Tip。3D 列名使用 `wrist_x` 等，四元数使用 `wrist_qx` 等，2D 使用 `wrist_u/wrist_v`，与仓库规范一致。只统一名称，不重排数据或变换世界坐标。四元数顺序为 xyzw，原值不归一化、不变换。IMU exports acceleration and angular velocity only; orientation, covariance and hand track_id are omitted. 源 SLAM 跟踪主体与外参未经确认。

`calibrations/camera_info.json` 保留两相机 CameraInfo 的 K/D/R/P、畸变模型、分辨率、frame_id；参数发生变化时保存变更时间。不从 P 推断外参，也不进行校正或将视频标记为已校正。

标注分两阶段处理。转换阶段保留 `annotation/annotation.json` 原文，生成 `annotation/episode_000000_source.json`：保留完整源标注、原始字段、片段索引和纳秒时间，只计算相对于首张左图的秒数，不翻译、不推断物体角色、不拆分动作。不再使用固定译文词表生成最终英文标注。

第二阶段在 `argus` 仓库运行 `python -m review.archebase.run`，调用模型做英文翻译与语义清洗，并由本地代码按 L3 schema 组装结果。模型处理 action/text/object/destination，时间区间和片段对应关系由程序固定，未知的可见性、进度、场景等保持为空。模型的请求、响应与回执单独保存。具体用法见 `argus/review/archebase/README.md`。

已有数据可仅重建中间标注，不重跑视频转换：

```powershell
python format_convert/archebase/annotations.py --dataset D:/data/archebase/186460_lerobot
```

2026-11-06 实测 `186460`：双目各 5,045 条视频消息，IMU 111,540 条，手部/SLAM 各 5,045 条，共 12 个输出特征。

2D validity masks (`observation.hand_{left,right}_points_2d_valid`) are omitted.
