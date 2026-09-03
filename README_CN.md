# mecta

`mecta` 是一个面向约束时间轴生成实验的独立代码仓库，包含 CREST 与
WCEP-CTG 两个经过格式统一的数据集、数据读取、事件聚类、Stage-II 监督构造、
交叉编码器训练、语义分数融合、时间轴解码与评测代码。

本仓库默认提供低显存的 `reference-event input` 运行方式：不训练或加载
Llama/QLoRA，而是将每个数据划分中的 reference events 转换为标准 Mention
输入。训练划分会确定性加入来自其他实体的 reference-event 干扰项，由原有的
全约束可靠负例筛选逻辑决定哪些样本可以作为负例。

开发集和测试集使用了对应划分的 reference events，因此该运行方式衡量的是
聚类、约束分配、排序和解码能力，不是端到端的文档事件抽取能力。运行产生的
元数据会保留 `uses_partition_references: true`，便于复核实验口径。

## 目录

```text
configs/                 CREST 与 WCEP-CTG 配置
dataset/                 两个数据集
mecta/                   核心 Python 包
scripts/                 各阶段命令行入口
tests/                   离线单元测试
REFERENCE_INPUT.md       reference-event 输入模式说明
requirements.txt         Python 依赖
run_all.sh               单数据集完整运行入口
```

实验输出默认写入 `runs/`，该目录被 Git 忽略，仓库中不包含本地实验结果、
检查点或缓存。

## 数据集

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

两个数据集均由 `mecta.data.DatasetReader` 通过统一接口读取。数据的继续分发和
使用应遵循其原始来源对应的许可与条款。

## 环境

```bash
python -m pip install -r requirements.txt
```

配置默认使用本地 GTE 与 MiniLM 模型。运行前请在
`configs/crest.yaml` 和 `configs/wcep_ctg.yaml` 中调整模型路径。
`--reference-input` 模式不会读取配置中的 Llama 基座模型。

## 运行

CREST：

```bash
PYTHON_BIN=python GPU_INDEX=0 ./run_all.sh crest runs/crest_reference_input
```

WCEP-CTG：

```bash
PYTHON_BIN=python GPU_INDEX=0 ./run_all.sh wcep_ctg runs/wcep_ctg_reference_input
```

Windows PowerShell 示例：

```powershell
python scripts/run_pipeline.py `
  --config configs/crest.yaml `
  --run-dir runs/crest_reference_input `
  --device cuda:0 `
  --reference-input
```

只进行数据、划分和路径预检：

```bash
python scripts/run_pipeline.py \
  --config configs/crest.yaml \
  --check-only \
  --reference-input
```

## 测试

```bash
python -m pytest -q
```

## 流水线

在 `--reference-input` 模式下，`prepare_stage1` 与 `train_stage1` 会被跳过：

1. 将 train/development/test reference events 写成 Mention 数据；
2. 对三个划分分别执行同日完整链接聚类；
3. 使用训练集 reference timelines 构造 Stage-II 正例和可靠负例；
4. 训练 MiniLM 交叉编码器并在开发集选择轮次；
5. 融合交叉编码器分数与 GTE 直接语义分数；
6. 按时间轴预算解码测试候选并评测；
