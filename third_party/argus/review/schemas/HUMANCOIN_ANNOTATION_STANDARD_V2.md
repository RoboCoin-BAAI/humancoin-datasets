# HumanCOIN 标注标准 2

日期：2026-11-18。目的：让新标注的场景、物品、动词和任务名称使用一致粒度，便于按 **场景 → 任务名称** 归组。

本标准与标准 1 并存，显式选择，不自动迁移旧标注，也不修改正在运行的历史清洗流程。JSON 结构仍兼容现有 `schema_version: "1.0"`；“标准 2”表示内容规范，记录于结构化回执，不向标注增加一套归并字段。

## 规范文件

| 内容 | 文件 | 规则 |
|---|---|---|
| 场景 | [HUMANCOIN_SCENES_V2.md](HUMANCOIN_SCENES_V2.md)、[humancoin_scenes_v2.json](humancoin_scenes_v2.json) | 6 个一级分类、当前 38 个二级推荐值；只有商业服务允许新增二级分类 |
| 物品 | [humancoin_objects_v2.json](humancoin_objects_v2.json) | 参考 Excel 的 636 个有效词条，简洁英文类别名，允许忠实的表外名词 |
| 动作 | [humancoin_actions_v2.json](humancoin_actions_v2.json) | Subtask_100 实际 99 个有效动词，Atomic_50 实际 46 个有效动词，排除数字尾注 110 |
| 输出 Schema | [episode_structured.standard2.schema.json](episode_structured.standard2.schema.json) | 独立的标准 2 校验，不替换原 Schema |
| 实现 | `review/standard2.py` | 模型提示、名称及动词规范化、task_name 构造、跨字段校验 |

## 场景：category / area

- 一级固定：`household`、`industrial_production`、`commercial_service`、`professional_work`、`public_space`、`other`。
- 非商业一级分类的二级枚举固定，不能新增。
- 商业服务优先使用已有二级分类，确实无法覆盖时可新增英文 `snake_case` 名称，并在回执中保留版本身份。
- 会议室统一使用 `professional_work / conference_center`；室内体育和室内羽毛球场地统一使用 `public_space / indoor_sports_venue`。
- 无依据时为 `other / other`，不能因出现杯子就默认厨房、出现货架就默认商店、出现工具就默认工厂。
- 历史光轮清洗中用户给出的家庭、卧室、客厅前提仅用于指定样本，不作为新数据的通用默认值。

## 物品：objects[].name

名称使用规范英文类别名，去掉冠词、数量、量词、颜色、大小、材质、形状、状态及品牌型号等修饰。优先自然单数，保留完整复合类别名。例外如 `scissors`、`glasses`、`pants` 使用其自然类别形式。

| 输入表达 | 规范名称 |
|---|---|
| two red cups / large blue cup | `cup` |
| large cardboard boxes | `box` |
| documents | `document` |
| strands of hair | `hair` |
| washing machine | `washing machine` |
| remote control | `remote control` |
| first aid kit | `first aid kit` |

同名不等于同一个实例，不合并或删除物体，不改变原对象数量和顺序；既有人物、身体部位也不因不在词表中而删除。属性依然独立保存，未知不猜测。已结构化输入的属性、位置和原始属性保持不变；原始 Argus 属性列表按现有颜色、材质、形状枚举分型。

## 动作：Subtask 与 Atomic

### 两类动词的用途

| 层级 | 用途 | 示例 |
|---|---|---|
| Subtask | 有明确操作目的的动作，作为 task_name 的候选动词 | `wipe`、`wash`、`fold`、`organize`、`sort`、`assemble`、`trim` |
| Atomic | 手部动作和基本移动，用于时间线的动作标签 | `pick`、`grasp`、`move`、`place`、`hold`、`adjust` |

逐段 `subtask[].action` 优先从两个参考表中选择，不强行把 Atomic 动作转成推测出来的任务目的。没有等价词时允许忠实、简洁的表外基础动词，例如有明确编辫证据时的 `braid`，在回执 `nonreference_actions` 中记录。

### 规范与合并

- 使用基础动词，不包含对象名、副词、方向、次数或长短语。
- 已知屈折形归一化，如 `trimming → trim`、`closing → close`、`picked → pick`。
- 明确同义或重复执行变体归并：`retrim → trim`、`rearrange → arrange`、`reassemble → assemble`、`reattach → attach`、`reinsert → insert`、`pick_up → pick`、`inspect → check`、`tidy → organize`。
- 不机械剥离前缀：`remove` 与 `move` 不合并，`release`、`retract` 不做错误切词。
- 相反动作不合并：`open/close`、`fold/unfold`、`screw/unscrew`、`attach/detach`、`insert/remove`。
- `pick/grasp/hold/place` 保留各自含义。
- 新生成标准 2 时，每段 `action` 也采用规范动词；`actions.vocabulary` 必须等于按段落顺序遍历得到的去重并集，不能出现只在总词表里存在的标签。

**与历史清洗范围的区别**：此前批量清洗按用户要求只改 `actions.vocabulary`，不动子任务标签；标准 2 用于新生成或显式另存的整份结构化结果，因此会同步规范逐段标签。动作原文、时间边界和段落数量、顺序不变。

## 任务名称：task_name

### 固定形式

```text
<一个规范的 Subtask 动词> <一个规范的主物品类别名>
```

全部小写，用空格连接。**不是随意生成的摘要句，也不是最高频 Atomic 动作。** 模型返回临时的 `{verb, object_index}`，程序验证后使用对应物品的规范名称拼接；该临时结构不写入标注。

- 动词必须来自 Subtask 参考表，经过同义归一化；不能直接用仅属于 Atomic 的 `pick`、`place`、`move`、`adjust` 等作为任务名动词。
- 主物品必须引用本条标注 `objects` 中已有的一个规范名称，不能凭任务名补造新物体。
- 选择整段任务主要作用的对象，不选操作者、顾客或默认选择手中的工具。
- 不包含场景、角色、颜色、数量、大小、方向、耗时、完成状态、目的从句或多个并列动词。
- 不能确定整体任务目的或主物品时使用 `null`，不编造 `organize` 等目的来凑模板。

### 稳定命名口径

| 证据支持的主要目的 | task_name 示例 | 边界 |
|---|---|---|
| 用布擦拭桌面 | `wipe table` | 不泛化成 clean，也不选工具 cloth |
| 用水或清洁剂洗杯子 | `wash cup` | 与干擦区分 |
| 折叠毛巾 | `fold towel` | 不是频繁出现的 place towel |
| 整理文件并归置 | `organize document` | 不带 scattered、neatly 等修饰 |
| 按类别分拣瓶子 | `sort bottle` | 以类别划分为目的 |
| 将盒子摞成堆 | `stack box` | 以堆叠为目的 |
| 按空间布局摆杯子 | `arrange cup` | 与普通收纳 organize 区分 |
| 包装物品 | `pack <主物品>` | 主物品根据包装目标判断，不一律取容器 |
| 修剪头发 | `trim hair` | 不使用 retrim，不写 customer hair |
| 切断材料 | `cut <主物品>` | 与修边 trim 区分 |
| 用零件装成电机 | `assemble motor` | 不用零件 part 替代成品目标，除非证据仅支持零件 |
| 打开盒子是整个任务 | `open box` | 不推测为整理/包装 |

同场景、相同目的、相同主物品的颜色或数量变体会得到同名任务；目的不同的 `wash cup` 与 `arrange cup` 不合并。Schema 与业务校验约束形式及物品引用，任务目的是否判断正确仍依赖标注证据和复核。

## 按“场景—任务名称”归并

推荐使用三个独立字段组成联合键，禁止只按 task_name 全局归并：

```python
from review.standard2 import grouping_key

key = grouping_key(annotation)
# ('household', 'kitchen', 'wash cup')
# key is None：场景为 other 或 task_name 为 null，进入待复核，不自动合并。
```

推荐逻辑层级：

```text
category/
  area/
    task_name/
      episode A
      episode B
```

主物品使用类别粒度，多个同类实例不会引入数量后缀。`common_task` 等历史字段不作为本标准的联合键来源，也不会被偷偷覆盖。

## 使用方式

### 新视频：显式选择标准 2

在 Argus 仓库根目录运行：

```bash
python -m review.base.run \
  --data /path/to/video.mp4 \
  --output /path/to/new/episode_000000_structured.json \
  --model qwen3.7-plus \
  --annotation-standard 2
```

共享入口 `python -m review ... --annotation-standard 2` 同样支持。默认仍为标准 1；不会把旧的完成结果静默重算为标准 2。旧输出不满足标准 2 时应使用新输出位置，或明确选择已有的强制重跑机制。

### 已有标注或 Argus episode：另存标准 2

```bash
python -m review.standard2 \
  --input /path/to/episode_000000_structured.json \
  --output /path/to/standard2/episode_000000_structured.json \
  --model qwen3.7-plus
```

输入也可以是 `parse_ok=true` 的原始 Argus episode JSON。源文件不修改，输出路径必须不存在；`--dry-run` 仅检查准备信息，不调用模型。模型及网关复用 `review/qwen_endpoint.py`，校验失败沿用共享结构化重试机制，每次请求的计费与回执逻辑沿用现有实现。

该独立入口处理一份标注；数据集批处理可逐文件调用它，或在已接入的共享视频入口选择标准 2。专用 L3/简智旧文本适配器的默认流程没有自动切换，不能假定所有旧入口都已采用新标准。

### 校验

```python
from review.standard2 import validate_final
validate_final(annotation)
```

除了独立 Schema，还检查场景配对、非商业二级限制、名称规范、逐段动作与总词表一致性，以及任务名的 Subtask 动词和主物品引用。标准及词表的身份写入外部回执，用于防止不同标准的缓存混用。

## 实施与验证状态

本地已新增标准 2 规则、机器词表、独立 Schema 和结构化入口。标准 1 文件与默认规则保留。37 项标准 2、旧结构化、旧场景和基础入口测试通过；真实简智标注的无模型调用准备预检通过。未实际调用新标准的模型 API，未迁移已有数据，也未同步覆盖服务器上正在运行的 Argus 代码。
