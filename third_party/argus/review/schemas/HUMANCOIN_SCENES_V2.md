# HumanCOIN 场景分类词表 V2

更新日期：2026-11-18。

基础分类来源：`D:/data/annotation_statistics/理想/HumanCOIN_Sence.xlsx`。此前临时新增的分类已按用户要求归并到原表分类，目前不额外保留新增二级分类。

一级分类 `category` 固定为 **6 类**（含 `other`）；二级分类 `area` 当前共 **38 个推荐值**（与原表一致，含 `other`）。仅商业服务 `commercial_service` 允许按需新增二级分类，其他一级分类的二级枚举固定。

## 一级分类

| category | 中文 | 当前二级分类数量 |
|---|---|---:|
| `household` | 家庭 | 9 |
| `industrial_production` | 工业生产 | 6 |
| `commercial_service` | 商业服务 | 10 |
| `professional_work` | 专业工作 | 5 |
| `public_space` | 公共场所 | 7 |
| `other` | 其他 / 待识别 | 1 |

`public_space` 同时涵盖室内和户外公共场所。V2 不再使用 `public_outdoor`、`leisure_social`、`unknown` 作为一级分类；旧数据迁移时需要结合二级分类及描述判断。

## 二级分类

### 家庭：household

| area | 中文 |
|---|---|
| `kitchen` | 厨房 |
| `living_room` | 客厅 |
| `dining_room` | 家庭餐厅 |
| `bedroom` | 卧室 |
| `study_room` | 书房 |
| `bathroom` | 卫生间 |
| `storage_room` | 储物间 |
| `laundry_area` | 洗衣区 |
| `balcony` | 阳台 |

### 工业生产：industrial_production

| area | 中文 |
|---|---|
| `factory_workshop` | 工厂车间 |
| `workroom` | 工作间 |
| `industrial_warehouse` | 工业仓库 |
| `equipment_room` | 设备机房 |
| `testing_lab` | 实验测试室 |
| `construction_site` | 建筑工地 |

### 商业服务：commercial_service

| area | 中文 | 来源 |
|---|---|---|
| `retail_store` | 超市 / 零售店 | 原表 |
| `restaurant` | 餐饮店 | 原表 |
| `beverage_shop` | 饮品店 | 原表 |
| `logistics_facility` | 仓储物流场所 | 原表 |
| `hotel` | 酒店 | 原表 |
| `hair_salon` | 理发店 | 原表 |
| `beauty_salon` | 美容店 | 原表 |
| `entertainment_venue` | 游戏娱乐场所 | 原表 |
| `pet_shop` | 宠物店 | 原表 |
| `performance_venue` | 演出场馆 | 原表 |

### 专业工作：professional_work

| area | 中文 | 来源 |
|---|---|---|
| `office` | 办公室 | 原表 |
| `classroom` | 教室 | 原表 |
| `laboratory` | 实验室 | 原表 |
| `healthcare_facility` | 医疗护理场所 | 原表 |
| `conference_center` | 会议中心 | 原表 |

普通会议室与会议中心统一使用 `conference_center`（会议中心），不再单列 `meeting_room`。

### 公共场所：public_space

| area | 中文 | 来源 |
|---|---|---|
| `park_garden` | 公园 / 花园 | 原表 |
| `street` | 城市街道 | 原表 |
| `public_facility` | 公共设施场所 | 原表 |
| `transportation` | 交通场所 | 原表 |
| `farm` | 农场 | 原表 |
| `sports_area` | 户外运动场地 | 原表 |
| `indoor_sports_venue` | 室内体育馆 | 原表 |

本轮羽毛球场地按用户确认归入 `indoor_sports_venue`，不再单列 `badminton_court`。

### 其他：other

| category | area | 中文 |
|---|---|---|
| `other` | `other` | 其他 / 待识别 |

## 分类规则

1. 一级分类只能使用上述 6 个枚举，不新增一级分类。
2. 二级分类优先复用本表。仅 `commercial_service` 在明确商业服务场景且已有枚举无法覆盖时，允许新增简洁的英文 `snake_case` 名称。家庭、工业生产、专业工作、公共场所及其他均禁止新增二级分类。同一二级枚举不跨一级分类使用。
3. 分类优先参考明确的场所信息，其次结合任务描述判断；用户提供的拍摄场所信息优先于模型推断。
4. 清洗现有光轮标注时，参考 `task_description` 和 `scene_classification.original_description`。不能只按旧标签机械替换，也不能把操作对象直接当成场所证据。
5. 依据不足时使用 `other / other`，待单独复核。不使用 `unknown` 或 `null` 兜底，不为无法判断的样本强行新增二级分类。
6. 按已确认的清洗范围，只修改原 JSON 中已有的 `scene_classification.category`、`scene_classification.area`，保留描述及其他字段，不新增字段。分组、判断依据和复核清单另存。
7. 基于任务推断的分类描述功能活动场景，不代表已通过视频核实了实际拍摄房间。

## 光轮清洗中已确认的归并

下表记录本轮光轮样本的修正口径。涉及卧室、客厅和室内体育馆的归并基于用户提供的场所经验，不能无条件推广到其他来源的数据。

| 本轮曾使用的临时 area | 当前 category | 当前 area |
|---|---|---|
| `dressing_area` | `household` | `bedroom` |
| `home_fitness_area` | `household` | `living_room` |
| `home_recreation_area` | `household` | `living_room` |
| `pet_care_area` | `household` | `living_room` |
| `shoe_care_area` | `household` | `living_room` |
| `food_service_area` | `commercial_service` | `restaurant` |
| `sports_equipment_area` | `public_space` | `indoor_sports_venue` |
| `badminton_court` | `public_space` | `indoor_sports_venue` |
| `meeting_room` | `professional_work` | `conference_center` |
| `massage_room` | `professional_work` | `conference_center` |

`massage_room` 归入会议中心是用户指定的数据归并口径，不表示按摩服务间与会议中心在通常语义上等价。后续不再生成该临时 area。

上述临时 area 不计入本版推荐词表。原 other 样本后续按用户要求以家庭为前提进行房间复分，这一家庭前提同样仅适用于指定样本，不能作为通用分类默认值。

## 与 V1 及程序规则的关系

本文件是新版分类的文档记录，V1 文档保留用于追溯。本次没有修改 `humancoin_scenes_v1.json`、`episode_structured.schema.json` 或程序提示词；相关程序不会因新增本文件自动采用 V2。

后续接入程序时，需要同步更新机器词表、Schema 和提示词，处理例如 `dining_area → dining_room`、`household_storage → storage_room` 等变更，并按完整的 category/area 配对校验。`leisure_social` 等旧分类涉及拆分归属，需结合描述判断，不能一律映射到单个新一级分类。
