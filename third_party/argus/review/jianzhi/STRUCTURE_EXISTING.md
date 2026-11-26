# 简智原始标注结构化

当前清理规则：该条数据所有 episode 结构化成功并通过校验，且源文件未改变后，删除 meta/annotations.json；任一失败或 dry-run 均保留。源已删除时可校验并复用对应最终结果，但不能 --force 重新调用模型，除非重新提供源标注。

运行文件位置：锁和模型请求收据位于系统临时目录 `argus-jianzhi-text/<输出路径哈希>/`，数据集 annotation 目录只发布最终 structured.json。旧版本的隐藏锁和请求收据在下次处理时迁出，不作为数据发布。

```powershell
python -m review.jianzhi.structure_existing --source D:\data\jianzhi\001301e26b8c4ace844c7b2d12eacf9d --dry-run
```

去掉 --dry-run 执行文本模型调用。默认 qwen3.7-plus，可用 --model 指定；--workers 默认 1，支持批次目录。密钥与 endpoint 使用 review.qwen_endpoint 的共享配置。--force 显式重算，有额外模型费用。

只读取 meta/annotations.json，原始文件不改。按 episode_index 对应 episode；保留最细粒度动作段的时间及 fine_label，原文成功标记保存在 notes 中。Protobuf 省略的首段零起点由父段起点或前段终点补齐。模型只补任务名、动作标签、原文提及的实体及功能场景，不读取视频、不重分段、不猜视觉属性。

Schema 允许 arm/contribution/progress/hands_visible 为 null，用于文本中没有证据的情况。其他无法确认的内容留 null 或空数组。annotation_provenance 标明文本来源、SST 和证据范围，source 记录原始 JSON 的哈希。父任务的成功结果基于原始成功标记派生，非视觉验收，不推断完成时刻。

最终结果通过共享 JSON Schema 校验后原子保存为 annotation/episode_*_structured.json。没有 Argus 视频工作目录；保留小型隐藏请求收据。当前执行默认最多尝试 3 次（首次加 2 次重试），重试包含网络、超时、暂时性 HTTP 错误、不完整响应、JSON 解析和结构化/Schema 校验失败，等待 2 秒、4 秒。可用 ARGUS_STRUCTURED_MAX_ATTEMPTS=1..10 设置总尝试次数。401/403 等永久请求错误不重试。逐次持久化尝试状态；重试可能重复计费。最终失败保留收据和原有输出，后续启动仍需先检查旧收据，不会无限自动重试。已有输出必须为同来源文本结构化结果才能复用。

原 review.jianzhi.run 仍提供视频标注功能，但 roboego-hand-vis 的简智 pipeline 已切换到本入口。
