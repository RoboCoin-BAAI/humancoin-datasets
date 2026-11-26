# 视频重建

从视频生成相机坐标系中的三维关节、网格及运动参数，并输出原图尺寸的叠加视频。坐标采用 OpenCV 约定：X 向右、Y 向下、Z 向前，距离单位为米。

## 安装

需要 NVIDIA CUDA GPU、兼容的驱动、Python 3.11 及以上，以及系统命令 `ffmpeg`、`ffprobe`。先按运行环境安装 CUDA 版 PyTorch，再安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

全部推理权重及必要资产已包含在目录中，低秩残差已合并为冻结权重，无需另外下载。整个目录可直接搬移到其他机器。

## 运行

```bash
python infer.py input.mp4 --output outputs/result
python app.py --device cuda:0 --host 0.0.0.0 --port 7860
```

也可以使用 `./run_app.sh`。默认处理全片；输出目录必须不存在。

- 默认自动选择精度。匹配的 SM89 / PyTorch 2.13.0+cu130 环境及支持尺寸使用 FP8；A100 和其他环境使用 BF16 加速。匹配的 H100 环境使用独立的 FP8 调优参数。
- 匹配已验证软件环境的 A100 使用 BF16 数据搬运融合，保留原有精度、4 帧编码块和默认窗口 batch 1，并使用确定性卷积策略。其他环境继续使用标准 BF16 路径。
- 首次运行包含加载、编译和预热；缓存保存在 `.cache/`，按权重、源码、GPU 和软件环境隔离。页面分别显示推理速度与总耗时。
- `--precision bf16` 显式选择 BF16；`--no-compile` 关闭编译。显式选择不支持的 FP8 配置会报错。
- BF16 默认窗口 batch 为 1；显存充裕时可用 `--window-batch-size 4` 等设置提高吞吐。窗口固定为 81 帧，步长固定为 40 帧。
- 方形和竖屏默认中心裁剪到 480×480；宽屏默认中心裁剪到 672×480。`--resize-mode long_edge_pad` 对宽屏等比缩放并补边，保留完整画面。
- `--max-frames 85` 限制处理帧数；`--overlay mesh` 切换网格；`--threshold 0.5` 调整显示阈值；`--no-overlay` 只导出预测。
- `--global-consistency` 统一整段视频的手形和相机，原始预测保留。
- `--geometry-camera-json camera.json` 使用指定内参进行三维推理；可与全局一致化组合使用，详见下节。
- `--camera-json camera.json` 接收原视频像素坐标中的 3×3 内参，仅用于叠加显示，不改变三维预测；不能与全局一致化同时启用。
- 自动探测硬件渲染和视频编解码能力，不可用时使用 CPU。可选 CUDA 网格渲染依赖见 `requirements-render.txt`。

仅重绘已有结果：

```bash
python infer.py input.mp4 --rerender outputs/result --overlay mesh
```

界面支持视频上传及预览、推理、下载、仅重绘和释放模型显存。MOV 等格式的浏览器预览可转换为 MP4，推理始终读取原始上传视频。

## 使用指定相机内参进行推理

默认不提供推理内参时，程序使用模型预测的相机。提供 `--geometry-camera-json` 后，会跳过相机内参预测头，用指定 K 生成射线，同时用于运动解码和三维平移求解。姿态、手形及深度仍由网络估计。

K 使用**原始输入视频**的像素坐标，程序会自动根据实际裁切、缩放和补边计算 `K_canvas = source_to_canvas @ K_source`，包括半像素缩放约定。不要提前手工缩放到模型画布。当前支持一段视频固定的 3×3 针孔内参：焦距为正、无 skew，最后一行为 `[0,0,1]`。画面须已去畸变；不接受逐帧 K、畸变参数或外参，也不自动去除鱼眼畸变。

`camera.json` 可直接保存 3×3 数组，也可使用 `K` 或 `intrinsics` 键：

```json
{"K": [[500, 0, 336], [0, 500, 240], [0, 0, 1]]}
```

示例数值仅说明格式，请替换为视频对应的真实标定值。

```bash
python infer.py input.mp4 --geometry-camera-json camera.json --output outputs/calibrated
python infer.py input.mp4 --geometry-camera-json camera.json --global-consistency --output outputs/calibrated_consistent
```

指定 K 与 `--global-consistency` 组合时，相机始终固定为给定 K，只统一手形并调整平移；根目录仍保存原始结果，`consistent/` 保存一致化结果。

Python 接口接受数组、上述字典或对应 JSON 字符串：

```python
from runtime.pipeline import Predictor

predictor = Predictor(device="cuda:0")
result = predictor.run(
    "input.mp4",
    "outputs/calibrated",
    geometry_intrinsics={"K": [[500, 0, 336], [0, 500, 240], [0, 0, 1]]},
    global_correction=True,
)
```

在 Gradio 的“更多设置”中填写“推理相机内参 K”，然后点击“运行推理”；留空即使用模型预测。修改推理 K 后必须重新运行推理。“仅更新叠加视频”继续使用既有结果中保存的几何相机，CLI 也禁止把 `--geometry-camera-json` 与 `--rerender` 组合。

“叠加显示内参 K”及 `--camera-json` 保留原有用法，只改变可视化。通常留空即可按实际推理相机显示；两种 K 同时提供时，推理使用推理 K、显示使用显示 K。启用全局一致化时，叠加显示 K 仍须留空。

底层 `runtime.accelerated_inference.infer_video(..., geometry_intrinsics=K_canvas)` 接收的是**处理后画布坐标**的 K，区别于上述 `Predictor.run` 的原视频坐标接口。普通 BF16 和支持环境的 FP8 均使用同一套标定射线；改变 K 数值不会重新加载模型。

## 输出

- `prediction.npz`：每帧的 `global_orient`、`hand_pose`、`betas`、`translation`、`joints_3d`、`vertices`、`anchors_2d`、存在及可见概率、相机内参、帧号、时间戳和重绘所需数据。手别维度顺序为左、右；姿态为轴角，`hand_pose` 为相对几何资产均值的残差。
- `metadata.json`：输入尺寸及坐标变换、处理设置、权重与结果校验和、阶段耗时。
- `overlay.mp4`：原视频尺寸的叠加预览，不保留音频。

启用全局一致化后，当前结果位于 `consistent/`，根目录保存原始预测。下载参数与页面所示结果对应。阈值仅影响显示，不裁掉预测帧。骨架颜色反映热图集中度，不能当作经过校准的准确率。

相机来源记录在 `geometry_camera_source`（`predicted` 或 `calibration`）中；指定 K 时还保存 `geometry_intrinsics_source`、`geometry_intrinsics_canvas`，并将 `decoder_and_pnp_use_calibration` 设为 `true`。NPZ 中的 `predicted_intrinsics` 是为兼容保留的历史字段名：指定 K 模式下，它逐帧保存实际使用的固定画布 K，并非模型重新预测的内参。`visualization_intrinsics` 单独记录仅用于绘制的 K。

第三方代码和资产说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
