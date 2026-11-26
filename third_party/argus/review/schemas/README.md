# 最终结构化标注 Schema v1

新增可显式选择的 [标注标准 2](HUMANCOIN_ANNOTATION_STANDARD_V2.md)：场景 V2、物品名、Subtask/Atomic 动词及“Subtask 动词 + 主物品”的 task_name。使用独立的 `episode_structured.standard2.schema.json` 和 `review/standard2.py`，不替换下述 V1 默认规则。

场景分类已统一为 [HumanCOIN V1](HUMANCOIN_SCENES_V1.md)：category 固定 8 个值，area 开放扩展，完整典型词表为 42 个（不含兜底）。旧场景枚举不自动迁移。

简智原始文本结构化支持：subtask 的 arm、contribution、progress、hands_visible 可为 null，表示原文没有证据；新增可选 annotation_provenance 来源说明。视频标注仍可输出已确认的具体值。

`episode_structured.schema.json` 使用 JSON Schema Draft 2020-12，定义 ego_head 流程（L1/L3/简智）的最终 `episode_*_structured.json`，不是模型原始输出格式。

以 L3-001-2026-10-26-13-21-17 的最终标注和当前结构化提取器为依据。顶层字段默认必填并拒绝未知字段；数组允许为空，未知值仅在明确允许的位置使用 null。动作使用开放的 snake_case 词汇，场景分类不限定为固定列表。

兼容现有生成行为：subtask.notes、事件和任务的 note、data_issues.end_s 为可选字段；state_changes.seen_in、capture_context、operator_mistakes 是可选扩展。capture_context 暂仅约束为对象，内部内容由采集配置定义。

验证示例（需要安装 jsonschema）：

```python
import json
from pathlib import Path
from jsonschema import Draft202012Validator

schema = json.loads(Path('review/schemas/episode_structured.schema.json').read_text(encoding='utf-8'))
data = json.loads(Path('episode_000000_structured.json').read_text(encoding='utf-8'))
Draft202012Validator.check_schema(schema)
Draft202012Validator(schema).validate(data)
```

Schema 检查字段、类型、枚举、数值范围及额外字段。以下关系需要额外业务校验：结束时间不早于开始时间、时间不超出视频、时间线覆盖、物体引用一致、动作词表与 subtask 一致、任务完成时间落在任务范围内、来源文件哈希与实际内容一致。英文语言和任务名最多 12 个词也是提示词/业务层要求。

本版提供规范文件，尚未自动接入标注生成和已有结果复用流程；不修改已有标注。
