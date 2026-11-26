# 结构化任务名称

`review/extract_dimensions.py` 在已有 Qwen 结构化请求中增加顶层 `task_name`，L1/L3 共用，不额外进行独立的命名请求。

模型主要参考 task_description 和 instruction_variants，并结合完整 subtask、tasks 和物品名称核对，输出英文“动词＋名词”的简短任务短语，如“Sort beverages”“Organize cutlery”“Weigh objects”。通常 2–6 个词，最多 12 个词、100 个字符；不罗列动作步骤，不把采集计划当成已发生事实，无法判断时返回 null。task_description、instruction_variants 和时间轴保留。

即使动作、属性与场景都能本地解析，也会为 task_name 调用一次结构化模型。缺少 task_name 的模型回复会报错，空串、换行、非字符串和超长值会被拒绝。英文及动词＋名词的语义要求由提示约束，程序只验证类型和长度。

已有结构化文件不会自动补写或重新计费，已有完成任务也不会因这个新增字段自动重跑。结构化处理器哈希已改变，旧文件与旧回执不应冒充新版输出。这里只更新新生成结果；补齐旧数据应另走显式结构化补全流程，避免重跑视频标注。
