# HumanCOIN 标注清洗统一规范 V3

日期：2026-11-22。适用于简智、光轮现有结构化标注的清洗及后续批次。以本规范作为本轮场景、动作、物品和任务名称的统一口径；历史文件保留追溯。新增规范文件不会自动改变现有脚本行为，执行前应确认脚本采用本版词表和规则。

## 1. 字段与修改范围

| 字段 | 规范 |
|---|---|
| `scene_classification.category`、`scene_classification.area` | 使用第 2 节场景分类 |
| `objects[].name` | 使用第 4 节物品命名规则 |
| `actions.vocabulary` | 优先使用第 3 节精选 99 个动作词，必要时允许有依据地新增，去重 |
| `task_name` | 一个主要手部操作动词＋一个主对象短语；动词必须属于精选 99 词，英文小写，以空格连接 |
| `common_task_name` | 根据主要动作、对象和目标选择既有 36 类之一，加 `_task`；不确定时用 `other_task` |

清洗已有标注时，只更新本次明确选定的字段。不更改子任务动作、动作描述、时间、物品数量和顺序、属性、原始描述；不向原标注增加解释、置信度或版本字段。词表外输入、判断依据、失败和待复核项写入独立报告。`common_task_name` 已有字段可更新；缺少该字段时按任务名称生成步骤添加。

## 2. 场景：category / area

基础枚举与中文名称见 [场景 V2](HUMANCOIN_SCENES_V2.md)，机器枚举见 [humancoin_scenes_v2.json](humancoin_scenes_v2.json)。基础共 6 个一级类别、38 个二级值（含 other）。

| category | 中文 | 允许的 area |
|---|---|---|
| `household` | 家庭 | kitchen, living_room, dining_room, bedroom, study_room, bathroom, storage_room, laundry_area, balcony |
| `industrial_production` | 工业生产 | factory_workshop, workroom, industrial_warehouse, equipment_room, testing_lab, construction_site |
| `commercial_service` | 商业服务 | retail_store, restaurant, beverage_shop, logistics_facility, hotel, hair_salon, beauty_salon, entertainment_venue, pet_shop, performance_venue |
| `professional_work` | 专业工作 | office, classroom, laboratory, healthcare_facility, conference_center |
| `public_space` | 公共场所 | park_garden, street, public_facility, transportation, farm, sports_area, indoor_sports_venue |
| `other` | 其他／待识别 | other |

分类规则：

1. 结合视频 10%、50%、90% 位置的抽帧、`task_description` 和 `scene_classification.original_description` 判断。实际场所证据和用户确认信息优先。
2. 不能因为出现杯子就判断厨房、出现工具就判断工厂，也不能因为某个动作就推断商业经营场所。
3. 一级分类固定；仅商业服务允许在已有场景确实无法覆盖时新增二级英文 `snake_case` 名称。新增项单独记录定义、依据和样本，后续复用同名，避免同义重复。其他类别禁止新增 area。
4. 必须按 category/area 配对校验；例如 `industrial_production / storage_room` 非法。
5. 普通会议室统一为 `professional_work / conference_center`；室内羽毛球场统一为 `public_space / indoor_sports_venue`。
6. 历史 `massage_room → conference_center` 是用户指定批次的归并规则；新数据按实际场所判断，不能将所有按摩场景自动归到会议中心。
7. 证据不足时使用 `other / other`，另存待复核清单；不得默认归家庭。

## 3. 动作：actions.vocabulary

依据 `HumanCOIN_任务_动作_原子动作词表_v0.5_精选99词.docx`。完整中英文词表见 [精选 99 词](HUMANCOIN_ACTIONS_V3.md)，机器词表见 [humancoin_actions_v3.json](humancoin_actions_v3.json)。这是动作词表，不是 36 类任务类别。

### 输出规则

- 尽可能使用 99 词表中的英文基础形式。没有语义等价的表内词时，允许新增有描述或画面依据的基础动词；不输出物品名、副词、短语、次数、方向或完成状态。
- 新增词只用于 `actions.vocabulary`，单独记录名称、含义、依据和样本；不自动扩充固定的 99 词参考表，也不成为 `task_name` 的合法动词。
- 例如 `carefully wiping table → wipe`、`retrim → trim`、`reassemble → assemble`、`inspecting → check`。只有语义等价时才替换。
- 按输入首次出现顺序去重。可合并同义变体，不能为凑齐词表凭空添加动作。
- `wipe`、`wash`、`rinse` 等在 99 词中均有独立含义，动作词表层保留差异；归到 `clean_task` 在任务类别层完成。
- 相反动作保留差异：open/close、fold/unfold、assemble/disassemble、plug/unplug。
- `brush` 需看对象：刷洗表面用 brush 或 scrub；梳理头发用 comb。`scrape` 刮除污物、打磨或刮痧需按描述判断，任务类别不能只看字面。
- `pick`、`place`、`hold`、`move` 不在本版 99 词中。不能机械换词：确实等价时，抓取可用 grasp，松手可用 release，抬起用 lift，放低用 lower；表内词无法忠实表达时，可作为新增动作保留，但禁止用于 `task_name` 的主要动词。
- 无法等价映射的输入不能静默删除或强行归类。有明确含义和证据时按新增动作记录；含义仍不确定时暂不写回该动作字段，另存待复核。`other`、`other_task` 都不是动作词。
- 此处仅清洗 `actions.vocabulary`，不自动同步 `subtask[].action`。历史子任务标签可能仍是旧词；报告需区分这两个范围。

18 个原子动作是 99 词的子集，可用于保留基本手部操作；任务名优先选择体现整段目的的动作，不因 grasp 高频就命名为 grasp。

## 4. 物品：objects[].name

使用简洁、稳定的英文物品类别名。参考 [现有物品词表](humancoin_objects_v2.json)；词表无法覆盖时允许使用有证据的常见物品名，不凭空创造物品。

### 去掉的内容

冠词、数量、量词、颜色、大小、材质、临时形状和状态、位置、序号、主观评价、品牌和型号等修饰。例如 red、large、small、beautiful、dirty、left、two、a piece of。

### 保留的内容

保留识别用途或类别所必需的复合名称，不能只按词性删除所有形容词：`remote control`、`washing machine`、`nail polish`、`hair dye`、`first aid kit`、`plastic wrap` 是完整物品名称。颜色和材质仅在构成稳定物品类别名时例外保留；普通描述性的材质应移除。

| 原名称 | 规范名称 | 说明 |
|---|---|---|
| two large red cups | cup | 去数量、大小、颜色，统一单数 |
| dirty wooden table | table | 去状态和描述性材质 |
| three cardboard boxes | box | 去数量和材质 |
| blue plastic bottle | bottle | 去颜色和材质 |
| small remote control | remote control | 保留完整类别 |
| black hair dye | hair dye | 去颜色，保留用途 |
| red nail polish | nail polish | 不简化为 polish |
| pieces of paper | paper | 去量词 |
| strands of hair | hair | 去量词 |
| scissors / glasses / pants | scissors / glasses / pants | 保留自然复数名称 |

其他规则：

1. 英文小写，单词之间用空格。可数名词优先单数；不可数名词和自然复数按正常英语保留。
2. 合并拼写、单复数及明确同义词；不能把不同功能物品强行归为 object、item、product，也不把具体物品自动提升为 furniture、cosmetics、tableware 等大类。
3. mask、glass、liquid 等多义名必须参考任务描述、原始描述和可用画面：面膜为 face mask，口罩为 face mask 或已有更明确的 medical mask，杯子为 drinking glass，玻璃材料为 glass；液体用途无法判断时保留 liquid，不猜成饮料或清洁液。
4. 已有 body、hair、face 等身体对象可保留。清洗不能删除人物或身体部位，也不能改变对象引用。
5. 仅替换 name，不改属性、ID、对象数量、顺序和子任务的 object/destination。即使多个对象清洗成同名，也不合并对象实例。

## 5. task_name 与 common_task_name

`task_name` 格式为 `<主要动词> <主对象短语>`，例如 `wipe table`、`comb hair`、`fold towel`。优先保证语义准确，任务归并由 `common_task_name` 承担，不为了压缩任务名称而丢失主要操作对象。

- 动词仍严格从精选 99 词中选择，只保留一个主要手部操作动词，不拼接多个动作。
- 主对象可根据 `task_description`、原始描述、子任务和视频画面提取，不要求出现在 `objects[].name` 中。已有清洗名称适用时优先复用，不适用时允许使用证据支持的对象短语。
- 允许忠实的集合名称，例如 `desktop item`、`medicine`、`clothing`。整理多种桌面物品时不任意挑选其中一个容器代表全部对象。
- 不强求主对象只有一个英文名词；保留用途和类别所需的限定词，例如 `medicine box`、`desktop item`、`hair dye`。不加入颜色、大小、数量、品牌、临时状态等无关修饰。
- 选择实际受作用的对象或内容，不默认选择工具或容器。将液体从一只瓶子倒入另一只瓶子时，主对象是液体，命名为 `pour liquid`。
- 任务名称中的集合名或描述中提取的对象不写回 `objects`，不新增对象实例，不改变已有对象和子任务引用。

| 任务证据 | task_name | 选择依据 |
|---|---|---|
| 将液体倒入另一只瓶子 | `pour liquid` | 使用实际转移的内容，不命名为 pour bottle |
| 整理多种桌面物品 | `organize desktop item` | 使用集合名称，不命名为 organize container |
| 整理多种药品 | `organize medicine` | 保留任务对象的共同类别 |
| 擦拭药盒 | `wipe medicine box` | 保留必要的用途限定词 |

`common_task_name` 按 [36 类配置](humancoin_task_categories_v0_5_cover.json) 输出：

```text
clean organize move assemble disassemble open close fold cut pack pour
apply secure replace operate check write groom stir discard store collect
fill peel dry polish sew shape massage remove manipulation processing
application interaction service cover
```

上述名称均加 `_task`。36 类是闭集，模型禁止新增类别；无法确定所属类别时允许直接使用 `other_task`，适用于新旧样本。`other_task` 是兜底值，不计入 36 类。不因新批次频次低于 5 就归 other。任务类别名 move、groom 等不一定属于 99 词，不能据此用于 `task_name` 的主要动词。

`task_name` 的动词严格限定为原始精选 99 词，不能使用 `actions.vocabulary` 的新增词。若没有能忠实表达整段目的的表内动词，单独记录待复核，不强行套用不等价词；`common_task_name` 可先定为 `other_task`。

| task_name | common_task_name | 分类理由 |
|---|---|---|
| wipe table / wash cup | clean_task | 清洁目标相同，具体方式仍保留在 task_name |
| comb hair / braid hair | groom_task | 头发梳理与造型 |
| scrape body | massage_task | 仅在描述明确为刮痧时 |
| scrape table | clean_task | 仅在描述明确为去污时 |
| cover sofa | cover_task | 覆盖保护物，采用后来确认的 cover 类 |
| close bottle | close_task | 盖瓶盖是封闭开口，不归 cover |
| wrap product | pack_task | 以包装为目的 |
| fold clothing | fold_task | 卷叠衣物按既有归并口径 |

文档中的动作默认类别只是候选映射，必须结合对象和目标判断；例如 scrape 不一律归 polish，press 不一律归 massage，roll 衣物不一律归 shape。`cover → pack` 的原文默认映射由本轮确认的 cover 类覆盖；包装仍归 pack。

不能依据现有描述确定主要动作或主物品时，单独记录待复核，保留原 task_name；不生成虚假的 organize object。新生成结果先独立保存，校验通过后只写回对应字段。

## 6. 写回前校验

1. 场景类别配对合法；商业新增项有独立记录。
2. 动作优先使用精选 99 词；新增词有独立记录和证据，无短语、重复或虚构动作。
3. 物品数和顺序不变，name 无无关修饰，属性及引用不变。
4. task_name 仅一个主要动词＋一个有证据支持的主对象短语，允许集合名称且不要求对象出现在 objects 中，动词必须属于精选 99 词；common_task_name 属于既有 36 类或不确定时使用的 other_task。
5. 比较写回前后 JSON，仅允许选定字段变化；失败样本不覆盖相应字段。

本轮只整理规范和独立参考词表，不执行服务器数据清洗，也不自动迁移旧脚本的词表。
