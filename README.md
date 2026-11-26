# HumanCoin Datasets

集中维护 HumanCoin 数据集的处理流程与入口。首批参考 `roboego-hand-vis` 和 `argus`。

| 路径 | 职责 |
| --- | --- |
| `humancoin_datasets/` | HumanCoin 数据处理功能及各阶段入口 |
| `scripts/`、`conf/` | 批处理入口、标定配置及隐私检测模型 |
| [`third_party/`](third_party/README.md) | 引入并维护的外部项目源码 |

## 已迁入的处理功能

- 接入与发布：`download_data`、`build_json`、`upload_data`。
- 质检与隐私：`lerobot_headband_integrity`、`video_privacy_blur`。
- 双目与推理：`lerobot_stereo_rectification`、`lerobot_fast_foundation_depth`、`lerobot_hand_pose`、`lerobot_vio`。

在仓库根目录可通过 `humancoin_datasets` 模块运行，例如：

```powershell
python -m humancoin_datasets.lerobot_fast_foundation_depth.L1.run --help
python -m humancoin_datasets.lerobot_hand_pose.run --help
python -m humancoin_datasets.lerobot_vio.L1.run --help
```

各模块 README 记录输入、输出和对应参数。深度和手部推理通过参数指定 `third_party/Fast-FoundationStereo` 和 `third_party/ego_hands_pose/video_reconstruction` 及所需模型权重，VIO 默认查找 `third_party/open_vins`。OpenVINS 仍需按对应平台构建。使用 KS3 或发布入口时，按需提供客户端、凭据和目标配置。批处理 YAML 中的服务器路径需要按运行环境填写。

L1 下载入口使用 POSIX 文件锁 `fcntl`，请在 Linux 或 WSL 环境运行。
