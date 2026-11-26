# 简智去畸变左目标注

简智入口只负责视频发现和输出路径，标注、结构化、最终 JSON Schema 校验及清理统一调用 `review.base.run`。默认模型为 `qwen3.7-plus`；成功后删除系统临时目录中的 `argus-base/<输入及输出路径哈希>/` 工作目录，失败保留。支持 `--keep-work` 显式保留工作目录。已有最终 JSON 也须通过 Schema 校验才能复用。

复用 L1 的 `review` / `ego_head` 视频语义标注流程。只读取
`videos/chunk-*/observation.images.head_left_rectified/episode_*.mp4`，
不读取手部 points、MANO，也不回退到原始视频。不需要 L1 的 cp_ 目录名。

在 argus 仓库根目录执行：

```bash
python -m review.jianzhi.run \
  --source /share/project/liuyou/humancoin/data/jianzhi/batch_0000001 \
  --workers 4 --no-board \
  --model qwen3.7-plus --reasoning none --sample-fps 2 \
  --max-tokens 16384 --concurrency 1 --cap 0
```

沿用 L1 的模型环境设置，例如 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`。
预算按每条视频计算，并行批次的总费用会累加，估算预算不是服务商硬扣费上限。

`--source` 支持单条数据或批次树；也可使用 `--data` 指定一条去畸变左目 MP4。
其余标注选项转发给共享 review，包括 `--free`、`--force`、`--dataset-note`。
`--out`、`--rig`、`--dataset` 由入口管理。

仅查看输入与命令（不创建文件、不调用模型）：

```powershell
python -m review.jianzhi.run `
  --source C:\Users\22563\Desktop\data\jianzhi\001301e26b8c4ace844c7b2d12eacf9d `
  --dry-run --no-board --model qwen3.7-plus --sample-fps 2
```

共享流程的准备及费用预览可使用 `--free`（会写准备文件，不调用标注模型）。

最终结构化 JSON 保存到：

```text
<session>/annotation/episode_000000_structured.json
```

使用与 L1/L3 相同的结构化提取器。默认模型为 `qwen3.7-plus`，默认关闭 board；仅接受能够触发共享结构化流程的 Qwen 模型。

Argus 工作文件放到系统临时目录 `argus-jianzhi/<视频路径哈希>/`，数据集内不创建 argus 目录。完整结果通过共享收据校验后原子写入最终 JSON，再清理临时工作目录。失败或 `--free` 时保留临时工作文件供诊断和续跑。

默认复用已存在的有效结构化 JSON，`--force` 归档临时工作并重算，成功后替换最终 JSON。不同 chunk 出现同名 episode 时会拒绝批量执行，防止最终文件冲突。
这是视频语义标注，输出动作、物体和场景等信息，不生成手部二维/三维关键点。
