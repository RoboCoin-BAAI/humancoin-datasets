# Archebase 源标注模型清洗

两阶段处理，不使用逐句硬编码译文：

1. `Multi-Frequency-LeRobot/format_convert/archebase/annotations.py` 将原始中文标注转换为 `episode_000000_source.json`。保留完整原文、原始字段和每段索引，只通过整数纳秒相减计算相对秒数。
2. 本模块调用文本模型做翻译和语义清洗，确定动作词、操作物、目标和参考物，再在本地组装成 L3 structured schema。

模型不重新标视频，不重新估计时间，也不生成可见性、穿戴、进度或场景事实。保持各有效段一一对应，不切分、不合并；源“无效”段记入 data_issues。不将 referenceObjects 当作 destination。描述与字段冲突由模型说明并写入 data_issues。

在 Multi-Frequency-LeRobot 仓库中，已有数据可只重建中间标注：

```powershell
python format_convert/archebase/annotations.py --dataset D:/data/archebase/186460_lerobot
```

在 argus 仓库中，先准备请求，不调用 API、不需要密钥：

```powershell
python -m review.archebase.run `
  --input D:/data/archebase/186460_lerobot/annotation/episode_000000_source.json `
  --out D:/data/archebase/186460_lerobot/annotation/episode_000000_structured_llm.json `
  --dry-run
```

正式运行使用相同命令，去掉 `--dry-run`。默认 `--model qwen3.6-plus`，复用 `review.qwen_endpoint` 的 XMSXB 配置（`XMSXB_API_KEY`、可选 `XMSXB_BASE_URL`）；其他已配置 Qwen 路由沿用 DashScope 设置。新输出使用 `_structured_llm.json`，避免误覆盖旧的手工词表结果。现有目标无匹配调用回执时拒绝覆盖。

输出包括去密钥的请求文件、完整模型响应、最终结构化标注及 `.claim.json` 回执（输入与处理代码哈希、模型、usage、输出哈希）。网络调用前持久化 claim，只发起一次请求；超时或未知结果不会自动付费重试。校验失败保留原始响应，不发布最终文件。可用 `--response-file <已保存响应>` 配合新输出路径进行离线校验，不再次调用模型。

校验有效段完整性、重复索引、原始时间计算、手别、英文文本、动作 snake_case 和对象字段结构。最终时间从中间格式复制，不接受模型返回的时间。不能仅靠结构校验保证翻译的语义正确，仍应抽查输出。
