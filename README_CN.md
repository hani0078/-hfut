# PoolTLS (Shared-Pool Timeline Summarization)

English version: [README.md](README.md)

`PoolTLS` 提供 CREST 与 WCEP-CTG 上的完整约束时间轴生成流程：构造第一阶段
监督数据、训练 Llama QLoRA adapter、从文章生成事件、聚类候选、训练第二阶段
交叉编码器、在开发集选择参数，并评测测试集时间轴。`run_all.sh` 与
`scripts/run_pipeline.py` 均运行这一完整流程。

## 目录

```text
configs/                 CREST 与 WCEP-CTG 配置
dataset/                 两个数据集
pooltls/                 核心 Python 包
scripts/                 各阶段命令行入口
tests/                   离线单元测试
requirements.txt         Python 依赖
run_all.sh               单数据集完整运行入口
```

预训练模型放在 `models/` 或配置指定的其他本地目录中。实验输出默认写入
`runs/`。仓库不包含模型权重、运行结果、训练检查点或缓存。

## 数据集与实验口径

```text
dataset/
├── crest_split/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── constraint_dict.json
└── WCEP-CTG-cleaned-20260831/
    ├── train/
    ├── validation/
    ├── test/
    └── statistics/
```

`pooltls.data.DatasetReader` 对外提供 `train`、`development` 和 `test` 三个划分，
其中 `development` 对应磁盘上的 `validation/`。数据的继续分发和使用应遵循
其原始来源对应的许可与条款。

第一阶段使用训练集文章及其金标准时间轴构造监督数据。训练得到的 adapter
随后分别根据各划分的文章正文和所请求的约束生成 Mention 记录，聚类与时间轴
构建使用这些生成事件作为候选文本。第二阶段使用训练集候选与训练集金标准
时间轴构造监督信号；开发集用于选择检查点和融合参数；最终指标由测试集预测
与测试集金标准时间轴比较得到。

当前解码配置为 `decoding.budget_source: reference_event_count`：每条所请求的
时间轴使用其金标准事件数作为输出长度预算，开发集和测试集也遵循该设置。
金标准事件文本用于监督和评测，不会被转换为生成候选文本。报告实验结果时，
应同时说明这一长度预算设定。

## 环境与模型

使用 Python 3.10 或 3.11，并准备支持 BF16 与 bitsandbytes NF4 训练的 CUDA
环境。在仓库根目录安装固定版本依赖：

```bash
python -m pip install -r requirements.txt
```

依赖包括 PyTorch、Transformers、PEFT、Accelerate、bitsandbytes 和 TILSE。
第一阶段使用 4-bit QLoRA 与 BF16 计算进行训练；生成阶段以 BF16 加载完整的
Llama 基座模型和已训练 adapter。请根据完整模型、文章长度和批大小准备显存。
第一阶段训练优先使用 FlashAttention 2，无法加载该后端时按配置回退到 PyTorch
SDPA；保留该回退配置时，`flash-attn` 为可选依赖。

准备包含权重、配置和 tokenizer 文件的完整本地模型目录。模型加载设置了
`local_files_only=True`，运行流程不会自动下载权重。配置中的以下默认路径相对
YAML 文件所在目录解析：

| 配置项 | 默认本地目录 | 用途 |
| --- | --- | --- |
| `paths.base_model` | `../models/Meta-Llama-3.1-8B-Instruct` | 第一阶段训练与文章事件生成 |
| `paths.gte_model` | `../models/gte-large` | 使用 GTE-large 进行语义检索、聚类、监督构造、负例选择与直接语义打分 |
| `paths.cross_encoder_model` | `../models/ms-marco-MiniLM-L-6-v2` | 第二阶段交叉编码器初始化 |

可以将模型放在仓库的 `models/` 下，也可以修改
[configs/crest.yaml](configs/crest.yaml) 和
[configs/wcep_ctg.yaml](configs/wcep_ctg.yaml) 中的模型路径，支持绝对路径。
实验记录应保留实际模型版本与配置改动。第一阶段训练与生成均保留完整文章输入，
超出配置中的 token 长度限制时会报错。

两个数据集均配置 GTE-large 作为冻结的语义编码器。第一阶段监督构造、候选聚类、
第二阶段监督构造与负例选择，以及开发集和测试集的直接语义打分，均读取
`paths.gte_model`。请将实际的 GTE-large 权重、配置与 tokenizer 文件放入
`models/gte-large`，或将该配置项指向已有的本地 GTE-large 目录。
目录名称本身不会改变其中存储的模型权重。

## 运行完整流程

以下命令均从仓库根目录执行。每次新实验使用新建或空的运行目录。完整运行会
训练两个阶段，并生成三个数据划分的文章事件。

CREST，使用 Bash：

```bash
PYTHON_BIN=python GPU_INDEX=0 bash run_all.sh crest runs/crest_full
```

WCEP-CTG，使用 Bash：

```bash
PYTHON_BIN=python GPU_INDEX=0 bash run_all.sh wcep_ctg runs/wcep_ctg_full
```

省略运行目录参数时，默认使用仓库根目录下的 `runs/<dataset>_full`。
`GPU_INDEX` 指定 Bash 入口可见的 GPU。

也可以直接调用 Python 入口；Windows PowerShell 示例：

```powershell
python scripts/run_pipeline.py `
  --config configs/crest.yaml `
  --run-dir runs/crest_full `
  --device cuda:0
```

运行 WCEP-CTG 时改用 `configs/wcep_ctg.yaml` 和独立的运行目录。PowerShell
示例同样需要配置好 CUDA 环境和本地模型。

训练前检查数据结构、划分和本地模型目录：

```bash
python scripts/run_pipeline.py --config configs/crest.yaml --check-only
python scripts/run_pipeline.py --config configs/wcep_ctg.yaml --check-only
```

预检会读取数据集并检查配置目录是否存在，不会加载模型权重、验证 GPU
兼容性或执行实验。

## 各阶段与续跑

`scripts/run_pipeline.py` 按以下顺序执行：

| 阶段 | 执行内容 |
| --- | --- |
| `prepare_stage1` | 将训练集金标准事件与训练文章对齐，生成完整文章的 SFT 记录。 |
| `train_stage1` | 训练 Llama QLoRA adapter，保存至 `models/stage1/final_adapter/`。 |
| `generate_train` | 使用该 adapter 从训练集文章生成事件 Mention。 |
| `generate_development` | 使用同一 adapter 从开发集文章生成事件 Mention。 |
| `generate_test` | 使用同一 adapter 从测试集文章生成事件 Mention。 |
| `cluster_all` | 分别对三个划分的同日 Mention 执行完整链接聚类。 |
| `prepare_stage2` | 构造训练正例，以及经过全约束筛选的可靠负例。 |
| `train_stage2` | 训练 MiniLM 实验组，并根据开发集指标选择检查点与融合参数。 |
| `select_development` | 将选定的第二阶段配置复制到 `selection/selected_config.json`。 |
| `score_test` | 为测试候选打分，融合交叉编码器与 GTE-large 分数，并按选定参数解码。 |
| `build_test_timelines` | 将已解码预测导出为 JSONL 和 CREST 时间轴目录格式。 |
| `evaluate_test` | 写出 TILSE ROUGE-1、ROUGE-2 和日期的精确率、召回率及 F1。 |

当前配置的融合权重网格仅包含交叉编码器权重 `0.50`。开发集选择范围由各
数据集的检查点和参数配置决定；测试集不参与检查点或融合权重选择。

需要先完成第一阶段训练、之后再继续时，可以使用独立运行目录：

```bash
python scripts/run_pipeline.py --config configs/crest.yaml --run-dir runs/crest_staged --device cuda:0 --stop-after train_stage1
python scripts/run_pipeline.py --config configs/crest.yaml --run-dir runs/crest_staged --device cuda:0 --resume --from-stage generate_train
```

`--resume` 会跳过预期输出文件已存在的阶段。`--from-stage` 必须与 `--resume`
一起使用，并要求此前阶段已经完成。续跑需要本流程写出的兼容
`run_manifest.json`。新实验使用新目录；继续已有实验时，应保持配置、数据集和
模型文件一致。

更新为 PoolTLS 后，请使用新的运行目录。运行标记现在记录
`method_name: PoolTLS`，检查点和选择结果文件的标识使用 `pooltls_` 前缀。
此前版本的运行目录不能直接使用本版本续跑。

流水线续跑不会自动恢复训练中断时的优化器状态。若第一阶段训练中断且已保存
检查点，可将下例的 `checkpoint-STEP` 替换为实际存在的检查点目录后执行：

```bash
python scripts/train_stage1.py --config configs/crest.yaml --train-file runs/crest_staged/stage1_data/train.jsonl --output-dir runs/crest_staged/models/stage1 --device cuda:0 --resume-from-checkpoint runs/crest_staged/models/stage1/checkpoints/checkpoint-STEP
```

最终 adapter 保存完成后，再使用流水线的 `--resume` 命令继续后续阶段。

## 输出与验证

每次运行都会保留配置副本、流程标记和阶段日志。运行目录中的主要产物如下：

| 路径 | 内容 |
| --- | --- |
| `stage1_data/` | 训练记录、文章对齐记录与监督数据统计 |
| `models/stage1/` | QLoRA 检查点、最终 adapter 和训练统计 |
| `mentions/<split>/` | 从文章生成的事件，以及 `_meta/` 中的解析统计 |
| `candidates/<split>/` | 聚类候选与聚类统计 |
| `stage2_data/train.jsonl` | 第二阶段监督训练样本 |
| `models/cross_encoder/` | 各实验组检查点与开发集选择结果 |
| `selection/selected_config.json` | 测试打分使用的检查点与融合参数 |
| `scores/test/` | 交叉编码器、直接语义和融合分数，以及解码预测 |
| `timelines/test_predictions.jsonl` | 最终测试集预测 |
| `timelines/crest/` | 每条时间轴单独存储的导出文件 |
| `evaluation/test_metrics.json` | 最终测试集评测指标 |
| `logs/` | 各阶段命令日志 |

执行离线测试：

```bash
python -m pytest -q
```

测试使用样例数据和 mock 检查软件行为。验证实验结果需要使用实际数据与模型
权重完成训练、生成和评测，并检查保存的中间产物与指标。
