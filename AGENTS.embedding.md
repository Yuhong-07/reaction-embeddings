# Metabolic Reaction Embedding Project

## 1. 项目目标

本项目建立一个代谢反应编码器，将每个代谢反应转换为固定长度数值向量：

```text
reaction record -> reaction encoder -> embedding vector [d]
all reactions -> embedding matrix [N_reactions, d]
```

默认输出维度 `d = 256`，必须允许通过配置修改。

必选输入仅包括：

- 反应物与产物的分子结构；
- 带符号的化学计量系数；
- 反应方向或可逆性。

可选输入包括：

- 反应类型；
- cofactor/cosubstrate 角色标记；
- EC number；
- atom-to-atom mapping 和键变化；
- 细胞区室；
- 物种、组织、培养基等生物学条件；
- 实验或计算得到的 flux。

当前阶段的唯一交付结果是每个反应的固定长度 embedding 及其 reaction ID 对照表。暂不进行反应产物生成、逆合成、性质预测或其他下游任务。

## 2. 核心建模决策

不要把反应仅当作一个 SMILES 文本序列。标准 reaction SMILES 不能可靠表达数值化学计量、可逆性、区室和 cofactor 角色，因此 reaction SMILES 只作为结构字段之一，结构化 reaction record 是权威输入。

主模型采用：

> 共享 D-MPNN 分子编码器 + stoichiometry-aware dual-side Set Transformer + 自监督表示学习

atom map 不能作为主模型必需输入。存在可靠 atom map 时，使用可选 CGR（Condensed Graph of Reaction）分支增强反应中心表示。

Flux 不属于反应的固有属性。必须将通用化学反应表示和条件相关表示分开：

```text
chemical_reaction_embedding
context_embedding
contextual_reaction_embedding
```

## 3. 统一数据格式

处理后的每个反应应使用以下逻辑结构。实际存储优先使用 Parquet；JSONL 可用于调试和数据交换。

```json
{
  "reaction_id": "RHEA:12345",
  "source_ids": {
    "rhea": ["RHEA:12345"],
    "metanetx": [],
    "modelseed": [],
    "bigg": []
  },
  "participants": [
    {
      "compound_id": "CHEBI:00000",
      "canonical_smiles": "...",
      "mapped_smiles": null,
      "coefficient": -2.0,
      "side": "reactant",
      "compartment": null,
      "role": null,
      "cofactor_role": null,
      "cofactor_missing": true
    },
    {
      "compound_id": "CHEBI:00001",
      "canonical_smiles": "...",
      "mapped_smiles": null,
      "coefficient": 1.0,
      "side": "product",
      "compartment": null,
      "role": null,
      "cofactor_role": null,
      "cofactor_missing": true
    }
  ],
  "reaction_smiles": "...>>...",
  "atom_mapped_reaction_smiles": null,
  "direction": "reversible",
  "reaction_type": null,
  "ec_numbers": null,
  "is_balanced": true,
  "context": null,
  "provenance": []
}
```

### 3.1 化学计量约定

- 反应物系数为负数；
- 产物系数为正数；
- 不要通过重复 SMILES 表示系数，例如 `2 ATP` 必须存为 `coefficient = -2.0`；
- 保留原始系数，同时可以向模型输入 `log1p(abs(coefficient))`；
- 聚合前可将整条反应约分为最小整数比，但必须保留未约分的原始字段和处理记录。

### 3.2 方向约定

统一枚举：

```text
left_to_right
right_to_left
reversible
undefined
```

数据源给出的物理/数据库方向和特定条件下的净 flux 方向不得混为同一字段。

对可逆反应，主化学嵌入应尽可能满足换向不变性：

```text
E_reversible = (E(R -> P) + E(P -> R)) / 2
```

如果下游任务需要有向表示，可同时保存正向和反向嵌入。

### 3.3 可选 Cofactor 约定

Cofactor annotation 不是生成 embedding 的必需输入。缺失 cofactor 信息时，反应仍必须进入数据集并正常生成 embedding。

必须区分：

1. 参与化学计量的 cosubstrate，例如 ATP、NADH、CoA；它们属于 reaction participants。
2. 不被消耗但为酶活性所必需的金属或辅基；它们属于 enzyme/context metadata。

不得仅凭一个短的人工 cofactor 列表删除 ATP、NAD(P)H、CoA、水或质子。参与化学计量的分子始终按普通 participant 编码；只有在可靠注释存在时，才额外加入可选的 `cofactor_role`。

推荐使用以下可空枚举：

```text
cosubstrate
catalytic_cofactor
metal
prosthetic_group
not_cofactor
unknown
```

缺失值必须表示为 `unknown/null + missing mask`，不能默认解释为 `not_cofactor`。

### 3.4 缺失数据

- 所有可选字段都要有显式 missing mask；
- 不要用零向量同时表示“真实值为零”和“字段缺失”；
- 模型必须能在没有 reaction type、cofactor、EC、atom map、flux 或 compartment 时运行；
- 每次数据导出都要报告各字段覆盖率。

## 4. 数据源及用途

没有单一开放数据集能完整、统一地提供所有字段，应按以下方式组合。

### 4.1 Rhea：主反应表

用途：

- 权威反应物、产物和化学计量；
- 方向和可逆性；
- 平衡反应；
- Rhea、ChEBI 和 EC 映射；
- reaction SMILES、transport flag 和反应层级。

Rhea 应作为 canonical reaction 的首选来源。

下载说明：https://www.rhea-db.org/help/download

### 4.2 EnzymeMap：atom map 数据

用途：

- atom-mapped reaction SMILES；
- 已平衡和清洗的酶反应；
- EC number；
- 训练反应中心和键变化任务。

EnzymeMap 不应作为 flux、区室或统一可逆性来源。与 Rhea 合并时优先依据标准化后的反应参与物、化学计量、EC 和外部 ID，而不是直接比较原始字符串。

数据：https://zenodo.org/records/7841781

### 4.3 UniProt：酶与非化学计量 cofactor

用途：

- Rhea reaction 与蛋白质的连接；
- EC number；
- enzyme-required cofactor、金属和辅基；
- 可选蛋白质序列或 protein embedding。

cofactor annotation 属于蛋白质，因此一次 reaction-to-UniProt join 可能产生多条 enzyme-specific records。通用 reaction record 中只保留聚合注释，原始映射必须另表保存。

字段说明：https://www.uniprot.org/help/return_fields

### 4.4 ModelSEED：建模和热力学信息

用途：

- 反应化学计量和可逆性补充；
- transport、compartment 和 pathway；
- metabolite structure；
- reaction balance/status；
- 热力学估计和建模相关元数据。

使用前必须依据 status 过滤不完整或不平衡反应，同时保留被过滤记录的统计。

数据：https://github.com/ModelSEED/ModelSEEDDatabase

### 4.5 BiGG/BiGGr：物种和网络环境

用途：

- organism-specific metabolic models；
- compartment；
- reaction bounds；
- gene-protein-reaction rules；
- transport、exchange、biomass 和普通代谢反应分类。

lower/upper bounds 是约束，不是测量 flux。由 FBA 求得的 flux 也必须标记为 `predicted_fba`，不能标记为实验值。

数据：https://www.biggr.org/data_access

### 4.6 MetaNetX：ID 对齐

用途：

- Rhea、ModelSEED、BiGG、MetaCyc 等数据库的 compound/reaction ID 对齐；
- compound SMILES 和交叉引用；
- 平衡与 transport 标记。

MetaNetX reference reactions 是 undirected，不应覆盖 Rhea 或模型中的方向字段。

说明：https://beta.metanetx.org/mnxdoc/mnxref.html

### 4.7 Flux 数据

实验 flux 来自单独的 fluxomics/MFA 数据集，例如 KiMoSys、MetaboLights 或论文附件。Flux 表至少包含：

```text
reaction_id
organism
tissue_or_cell_line
condition_id
flux_value
flux_unit
uncertainty
method: measured | 13C_MFA | inferred | predicted_fba
reference
```

不能将不同单位、不同碳源或不同归一化方式的 flux 直接合并。

## 5. 数据处理流程

按以下顺序构建数据：

1. 下载原始数据并记录版本、日期、URL、license 和 checksum。
2. 用 Rhea 建立 canonical reaction table。
3. 使用 ChEBI/MetaNetX 标准化 compound ID。
4. 使用 RDKit 解析、清洗和 canonicalize SMILES。
5. 解析反应参与物及带符号化学计量。
6. 验证元素和电荷平衡；不要静默修复。
7. 对齐 EnzymeMap atom mappings，并记录匹配置信度。
8. 连接 UniProt enzyme/cofactor metadata。
9. 连接 ModelSEED 和 BiGG 的 compartment、type、bounds 与 network context。
10. 单独建立 condition-level flux table。
11. 去重并冻结用于生成 embedding 的全量数据版本。

每一个自动修改反应的操作都必须保留：

```text
original_value
processed_value
transformation_name
software_version
confidence
```

### 5.1 SMILES 标准化

至少执行：

- RDKit parse validity；
- 盐和多组分处理记录；
- canonical isomeric SMILES；
- 电荷、同位素和立体信息保留；
- atom-map number 与 canonical SMILES 分开存储；
- 质子化状态不得在不同来源间静默替换。

### 5.2 数据去重

去重不能只使用 reaction SMILES 字符串。优先构造以下 canonical key：

```text
sorted[(compound_identity, signed_stoichiometry, compartment)]
+ direction_class
```

同时计算忽略质子、水、方向和 compartment 的宽松匹配键，用于发现可能重复项，但宽松匹配不得自动合并。

### 5.3 全量数据使用原则

完成标准化、质量过滤和去重后，所有有效反应都进入同一个 embedding corpus，用于自监督表示学习和最终矩阵导出。

方向变体、质子化变体和跨数据库副本应在去重阶段建立 equivalence group。默认每组保留一个 canonical reaction embedding；如需保留多个来源版本，必须共享同一个 `canonical_reaction_id`。

## 6. 模型架构

### 6.1 分子编码器

使用共享权重 D-MPNN：

```text
molecular graph G_i -> D-MPNN -> molecule embedding h_i
```

推荐默认值：

```yaml
molecule_hidden_dim: 256
molecule_message_passing_steps: 4
dropout: 0.1
readout: attention
```

原子特征至少包括：atomic number、formal charge、degree、aromaticity、hybridization、chirality 和 hydrogen count。

键特征至少包括：bond type、conjugation、ring membership 和 stereochemistry。

### 6.2 Participant token

对每个参与物构造：

```text
z_i = concat(
  h_i,
  side_embedding,
  stoichiometry_mlp(log1p(abs(nu_i))),
  optional_role_embedding,
  optional_cofactor_embedding,
  optional_compartment_embedding,
  missing_masks
)
```

`role_embedding`、`cofactor_embedding` 和 `compartment_embedding` 都是可选分支。字段缺失时必须使用独立的 missing embedding/mask，主模型仍只依赖分子结构、side、stoichiometry 和 direction 产生完整 embedding。

### 6.3 双侧集合编码器

分别编码反应两侧：

```text
h_R = SetTransformer({z_i | nu_i < 0})
h_P = SetTransformer({z_i | nu_i > 0})
h_delta = sum_i(nu_i * W * h_i)
```

Set Transformer 必须无视同一侧参与物的输入顺序。可以在 reactant/product set 之间增加 cross-attention，但 MVP 阶段不是必需项。

### 6.4 元数据融合

```text
h_rxn = MLP(concat(
  h_R,
  h_P,
  h_P - h_R,
  h_delta,
  direction_embedding,
  optional_reaction_type_embedding,
  optional_EC_embedding,
  missing_masks
))
```

EC 和 reaction type 仅作为可选增强字段。EC 使用层级编码：四级 EC 分别嵌入再组合，多 EC reaction 使用 set pooling。无这些字段时，模型必须保持相同输出维度。

### 6.5 可选 CGR 分支

有可靠 atom map 时：

```text
atom-mapped reaction -> CGR -> CGR-DMPNN -> h_cgr
```

CGR 特征应显式包含反应前后两套 atom/bond state。通过 missing-aware gate 融合：

```text
g = atom_map_available * sigmoid(MLP(...))
h_final = LayerNorm((1 - g) * h_rxn + g * Project(h_cgr))
```

没有 atom map 时，模型仍须产生完整结果。

### 6.6 可选代谢网络分支

独立反应编码器稳定后，才允许增加 metabolite-reaction bipartite graph 或 hypergraph GNN：

```text
metabolite nodes <-> reaction nodes
```

该分支学习 pathway/network context，不得替代化学反应编码器。最终应分别输出 standalone 与 network-aware embeddings。

## 7. 自监督训练目标

当前阶段不依赖下游标签。默认训练目标为：

```text
L_total =
  lambda_contrastive * L_contrastive
  + lambda_masked * L_masked_participant
  + lambda_stoich * L_stoichiometry
  + lambda_direction * L_direction
  + lambda_edit * L_atom_edit
```

任务说明：

- `L_contrastive`：同一反应的 randomized SMILES、来源变体和允许的方向变体作为正样本；
- `L_masked_participant`：mask 一个参与物后预测其 identity 或 embedding；
- `L_stoichiometry`：预测被 mask 的系数或系数区间；
- `L_direction`：预测 direction class；
- `L_atom_edit`：仅对可靠 atom-mapped reactions 计算。

所有 loss 都要按字段可用性 mask，不能因为 cofactor、reaction type、EC、atom map 或 context 缺失而删除整个样本。可选元数据默认只作为输入增强，不设置强制预测头。

## 8. Embedding 质量控制

当前阶段不以分类、回归或 retrieval 任务作为验收条件。必须执行以下与 embedding 生成直接相关的检查：

- 输出矩阵形状固定为 `[N_reactions, embedding_dim]`；
- 每一行能通过 reaction ID 唯一追溯；
- embedding 不得包含 NaN 或 Inf；
- 同一侧 participant 顺序打乱后，embedding 在数值容差内不变；
- 相同 canonical reaction 的不同 SMILES 表达应得到相近 embedding；
- reversible reaction 换向后满足定义好的对称性；
- 缺少全部可选字段时仍能生成 embedding；
- 批处理与单样本处理结果一致；
- 相同 checkpoint、配置和随机种子重复导出时结果可复现。

输出前必须生成质量报告：

```text
reaction_count
embedding_dim
field_coverage
failed_smiles_count
imbalanced_reaction_count
deduplication_count
nan_or_inf_count
embedding_norm_statistics
output_checksum
```

## 9. 推荐目录结构

```text
data/
  raw/                 # 原始文件，只读，不手工修改
  interim/             # 单一数据源标准化结果
  processed/           # 对齐、质量过滤和去重后的全量 embedding corpus
  manifests/           # URL、版本、license、checksum、覆盖率
configs/
  data/
  model/
  train/
src/
  data/
  chemistry/
  models/
  training/
  quality_control/
scripts/
tests/
artifacts/
  embeddings/
  checkpoints/
  reports/
```

## 10. 实施顺序

### Phase 1：数据准备

- 完成 Rhea parser 和统一 reaction schema；
- 完成 SMILES、stoichiometry、direction 验证；
- 完成去重、canonical reaction ID 和数据覆盖率报告；
- 冻结全量 embedding corpus 和 data manifest。

### Phase 2：最小深度模型

- D-MPNN 分子编码；
- reactant/product 分侧 sum 或 attention pooling；
- 加入必选的 stoichiometry、direction；
- 支持可选 reaction type、cofactor、EC 和 compartment 及其 missing mask；
- 输出 256 维 embedding；
- 完成全量矩阵、reaction ID 对照表和质量报告导出。

### Phase 3：完整反应编码器

- Set Transformer 和 cross-attention；
- 多任务对比预训练；
- EnzymeMap atom-map 对齐；
- optional CGR branch；
- 验证 canonical SMILES、participant 顺序和 reversible direction 一致性。

### Phase 4：生物学环境

- UniProt enzyme/cofactor；
- ModelSEED/BiGG compartment 和 network context；
- hypergraph/network-aware embedding；
- condition-specific flux branch。

## 11. Agent 工作规则

后续 agent 修改本项目时必须遵守：

1. 先检查数据 schema、配置和测试，不得假设 reaction SMILES 包含全部信息。
2. 数据处理必须可复现；禁止在 notebook 中留下唯一实现。
3. 原始数据只读，所有修改写入 interim/processed 并保留 provenance。
4. 新增字段必须同步更新 schema、coverage report、serialization 和测试。
5. 当前阶段只生成 embedding；不得擅自扩展为下游分类、回归或其他预测项目。
6. 不得把 flux bounds、FBA flux 和 experimental flux 混用。
7. 不得因 optional metadata 缺失而丢弃大部分反应。
8. 可选元数据只能作为 missing-aware 增强输入，不得成为反应能否生成 embedding 的前置条件。
9. 对跨数据库映射保留 `exact/high/medium/low` confidence，不自动接受模糊匹配。
10. 每次训练保存 config、git revision、data manifest、random seed、checkpoint 和 embedding quality report。
11. 优先实现 Phase 1/2 的端到端可运行版本，再增加 CGR、hypergraph 或 flux 分支。
12. 若实现选择与本文档冲突，应在代码或设计记录中解释原因和实验依据。

## 12. MVP 验收标准

MVP 完成需要同时满足：

- 能将至少一个公开数据源转换为统一 reaction schema；
- 每个反应物和产物都有明确 side 和 signed stoichiometry；
- 可以在没有 reaction type、cofactor、EC、atom map、compartment、context 和 flux 时运行；
- 能输出形状为 `[N_reactions, 256]` 的矩阵及 reaction ID 对照表；
- embedding 矩阵没有 NaN/Inf，并报告向量范数统计和文件 checksum；
- 同一侧 participant 顺序打乱后，embedding 数值在容差内不变；
- 可逆反应换向一致性有自动测试；
- 输出数据覆盖率、失败 SMILES、不平衡反应和去重统计；
- 训练和 embedding 导出可以通过配置文件和单一命令复现。
