# 物品名称和属性约束

统一词表与归一化实现：`review/object_policy.py`。适用于共享视频结构化、简智文本结构化、Archebase 文本清洗。以下为初版英文枚举；未知、无证据或词表外值使用 `null`，不猜测。

| 属性 | 枚举 |
| --- | --- |
| color | black, white, gray, red, orange, yellow, green, blue, purple, pink, brown, beige, cyan, magenta, gold, silver, transparent, multicolor |
| material | plastic, metal, wood, glass, ceramic, paper, cardboard, fabric, rubber, silicone, leather, stone, concrete, foam, composite |
| shape | round, square, rectangular, triangular, oval, cylindrical, spherical, conical, cuboid, flat, irregular |

名称使用简洁英文类别名，提示词要求去除颜色、数量、形状、材质、大小和冠词。程序清理已知修饰词和数字，但保留末尾类别名，避免误删 orange（水果）、glass（杯子）、box（箱子）。这是一层有限词表检查，不能替代完整语义判断，也不自动合并同名物体。未收录的名称修饰词仍需扩充规则。

例如 `two red round plastic cups` → 名称 `cups`，color=`red`、material=`plastic`、shape=`round`。名称中明确出现的枚举属性也参与提取；动作原文和时间线引用保持原有证据内容。

别名如 grey → gray、wooden → wood、cylinder → cylindrical。多种明确属性使用主值及 `additional` 数组，数组值同样受枚举约束。无法归一化的原属性保留在 `unclassified`，`original_attributes` 保留输入属性。最终 JSON Schema 限制主值和 additional，程序额外校验名称。

不批量迁移已有结果，不自动发起模型请求。已有结果在执行新版 Schema 校验时可能不兼容；旧完成记录的复用策略保持原样。需要更新历史数据时应单独制定离线迁移方案。
