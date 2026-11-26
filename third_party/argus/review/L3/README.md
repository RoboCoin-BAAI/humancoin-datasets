# L3 执行阶段标注

默认读取每个 episode 对应的 `meta/qc_reports.jsonl` 记录：

```text
qc_report.streams.marker.stats.markers.items[]
  name 或 tag == EXEC_START
  t_offset: 相对 RUN_START 的秒数
```

只采样 `t_offset` 及之后的矫正左目画面。视频本身不裁剪、不重新编码；送给模型的采样帧和解码失败后的替代帧均受此边界限制。输出任务、动作和事件时间保留原视频时钟，不从执行开始重新归零。默认执行到视频结束，不按 EXEC_END 截断。

示例 L3-001-2026-10-26-11-07-59 的 EXEC_START 为 10.193434000015259 秒；30 FPS 视频首个合格帧通常为约 10.2 秒。此前的指令展示与动作想象不参与标注。

要求矫正视频从本 episode 的 RUN_START 时间轴起点开始；不能把其他经过任意裁剪、重新归零的视频直接代入。缺少 qc_reports、缺少或重复 EXEC_START、无效偏移或偏移超出可用帧时，报错，不回退到整段标注。当前执行边界功能支持视频输入；需要自动分段的长视频会明确拒绝，尚未支持分段后的边界传播。

输入记录 `l3_input.json` 保存 QC 文件哈希和执行偏移。已有旧版全时段标注不会静默复用，需 `--force` 归档旧标注后重新调用模型。`--dry-run` 只读取标记并打印计划，不调用接口；视频范围验证在实际准备阶段完成。

服务器同步 `review/L3/run.py`、`review/__main__.py`、`label/episode.py`，以及相邻 roboego-hand-vis 仓库的 `scripts/L3/pipeline.yaml`。接口环境变量设置好后，在 roboego-hand-vis 根目录执行：

```bash
bash scripts/L3/pipeline_cpu.sh --batch-range 1:9 --workers 4 \
  --run-stages review --force-stages review
```

这会重做所选批次的标注，并保留原来的标注备份；不会重新跑上游计算，但需要有效的上游状态。不要同时启动另一个处理同批次的标注进程。
