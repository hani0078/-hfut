# mecta

Chinese version: [README_CN.md](README_CN.md)

`mecta` is a standalone repository for constrained timeline generation experiments. It includes the CREST and WCEP-CTG datasets in a unified format, together with code for data loading, event clustering, Stage-II supervision construction, cross-encoder training, semantic-score fusion, timeline decoding, and evaluation.

This repository provides a low-GPU-memory `reference-event input` mode by default. In this mode, Llama/QLoRA is neither trained nor loaded. Instead, the reference events from each data split are converted into standard Mention inputs. For the training split, deterministic distractors derived from the reference events of other entities are added, and the existing all-constraint reliable-negative filtering logic determines which examples can be used as negative samples.

The development and test splits use the reference events belonging to their respective splits. Consequently, this mode evaluates event clustering, constraint-aware assignment, ranking, and timeline decoding; it does not evaluate end-to-end event extraction from documents. Generated run metadata retains `uses_partition_references: true` so that the experimental setup remains explicit and auditable.

## Repository Structure

```text
configs/                 Configurations for CREST and WCEP-CTG
dataset/                 The two datasets
mecta/                   Core Python package
scripts/                 Command-line entry points for each stage
tests/                   Offline unit tests
REFERENCE_INPUT.md       Documentation for reference-event input mode
requirements.txt         Python dependencies
run_all.sh               End-to-end entry point for one dataset
```

Experiment outputs are written to `runs/` by default. This directory is ignored by Git, and the repository does not include local experiment results, checkpoints, or cache files.

## Datasets

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

Both datasets are loaded through a unified interface provided by `mecta.data.DatasetReader`. Any redistribution or use of the datasets must comply with the licenses and terms of their original sources.

## Environment Setup

```bash
python -m pip install -r requirements.txt
```

The default configurations use local GTE and MiniLM models. Before running the pipeline, update the model paths in `configs/crest.yaml` and `configs/wcep_ctg.yaml` as needed. The Llama base-model path in the configuration is not accessed when `--reference-input` mode is enabled.

## Running the Pipeline

CREST:

```bash
PYTHON_BIN=python GPU_INDEX=0 ./run_all.sh crest runs/crest_reference_input
```

WCEP-CTG:

```bash
PYTHON_BIN=python GPU_INDEX=0 ./run_all.sh wcep_ctg runs/wcep_ctg_reference_input
```

Windows PowerShell example:

```powershell
python scripts/run_pipeline.py `
  --config configs/crest.yaml `
  --run-dir runs/crest_reference_input `
  --device cuda:0 `
  --reference-input
```

To validate only the dataset structure, data splits, and configured paths:

```bash
python scripts/run_pipeline.py \
  --config configs/crest.yaml \
  --check-only \
  --reference-input
```

## Tests

```bash
python -m pytest -q
```

## Pipeline

When `--reference-input` mode is enabled, the `prepare_stage1` and `train_stage1` stages are skipped:

1. Convert the train, development, and test reference events into Mention records.
2. Perform same-day complete-linkage clustering independently for all three splits.
3. Construct Stage-II positive examples and reliable negative examples from the training reference timelines.
4. Train the MiniLM cross-encoder and select the training epoch on the development split.
5. Fuse the cross-encoder scores with the direct GTE semantic-similarity scores.
6. Decode the test candidates under the timeline budgets and evaluate the resulting timelines.
