# 通用 LeRobot readme 生成器

## 发布元数据结构

`meta/readme.json` 描述一个独立数据实例。README 和 `portal.yaml` 生成器消费该摘要，
加载器的特征定义仍由 `meta/info.json` 提供。下面约定适用于 base 生成器。

| 字段 | 类型 | 含义和来源 |
| --- | --- | --- |
| dataset_name | string | 发布集合名，包含层级、场景和任务等命名信息 |
| unique_task_name | string，仅 L1 | 实例发布目录名，包含源实例名以区分相同任务 |
| hand_capture_type | string | 当前 base 固定为 bare_hand |
| scene | object | type、category、name，name 对应标注的 area |
| task_name | string 或 null | 最终标注的具体任务名 |
| common_task_name | string 或 null | 最终标注的任务大类 |
| task_description | string 或 null | 最终标注的整段描述 |
| actions | string[] | 各 episode 动作词汇合并去重 |
| objects | object[] | 按清洗后 name 去重，每项仅包含 name |
| duration_s | number 或 null | 主帧数除以主帧率，单位秒 |
| frame_count | integer 或 null | master_index 行数，缺失时使用 info.total_frames |
| master_fps | number 或 null | info.fps |
| size_bytes | integer | 当前实例文件大小之和，排除 readme.json 及其写入临时文件 |
| file_count | integer | 与 size_bytes 相同范围的文件数 |
| cameras | object | 视频流计数、主相机和各流的规格 |
| modalities | object | 按层级允许且实际存在的数值数据流 |

### cameras

`count` 为 info.features 中 dtype=video 的数量，表示视频流数量。
原始和校正流同时存在时分别计数。L1 同时输出 color_count 和 depth_count。
`primary_camera` 为选定的校正左目特征名，`synchronization_master` 来自 info.master_feature。
`items` 每项包含 id、width、height、fps、sample_count，L1 另有 is_depth。
width/height 单位为像素。L1 的 sample_count 来自对应相机时间戳 Parquet 的行数。

### modalities

| 层级 | 允许的分组 |
| --- | --- |
| L1 | hand_point、head_pose、camera_pose、imu、eye_gaze |
| L2 | L1 分组，加 emg |
| L3 | L1 分组，加 emg、eeg |

不存在的分组省略。每组为 `{"streams": [...]}`，每条流包含：

- feature_key，与 info.features 和数据文件名一致。
- shape，特征维度数组。
- sample_count，实际 Parquet 行数。
- sample_rate_hz，特征 fps 或 sample_rate。
- coordinate_frame、quaternion_order，源特征声明时保留。

hand_point 包含供应方 raw_hand_*_points 和 hand_*_points。
camera_pose 包含 observation.camera_pose 及带相机位置限定的 *_camera_pose。
时间戳的秒制、坐标系及单位解释沿用 info 中的声明。

### 发布端读取关系

单实例 README 读取该实例的 readme.json 和 info.json。
L1 集合 README/portal.yaml 从各实例摘要汇总帧数、时长、文件数和体积；
集合目录名必须与 dataset_name 一致，实例目录名必须与 unique_task_name 一致。
实例 meta/readme.json 自身的大小和文件数在集合汇总时补计。
当前集合的传感器规格取首个实例的 info.features，尚未核对整个集合的规格一致性。

KS3 的 area_common_task_name 分组目录与发布集合名是两个命名用途，
准备发布集合时仍需按 dataset_name/unique_task_name 组织目录。


复用原光轮实现，输出字段和统计规则不变。无需模型调用，生成正式文件 `meta/readme.json`。读取视频容器帧数，不逐帧解码。

```powershell
python -m humancoin_datasets.build_json.base.build_readme --root D:\data\jianzhi\001301e26b8c4ace844c7b2d12eacf9d
```

也支持直接运行 `python humancoin_datasets/build_json/base/build_readme.py --root ...`。root 支持单数据集、batch 或包含 batch_* 的目录。`--level L1|L2|L3` 默认 L1。

通用信息来自 meta/info.json、meta/tasks.jsonl、master_index.parquet、实际视频与 Parquet、annotation/episode_*_structured.json。场景类型使用 HumanCOIN V1 对应规则；光轮旧入口继续转调本实现。

物品汇总各标注文件中的 objects[].name，按名称去重，仅输出 name，不输出属性。
任务大类 common_task_name 读取排序后第一个结构化标注文件的同名字段；缺失时为 null。

本次仅抽取共用实现：相机 count 仍为视频特征数量，包含校正派生流；MANO 尚未纳入模态摘要；不自动推断缺失坐标约定。原 L3 专用生成器及简智处理流水线未切换。写入使用同目录原子替换，成功后不保留临时文件。
