# PoolTLS (Shared-Pool Timeline Summarization)

Chinese version: [README_CN.md](README_CN.md)

`PoolTLS` implements the complete constrained timeline generation workflow for
CREST and WCEP-CTG: prepare Stage-I supervision, train a Llama QLoRA adapter,
generate events from articles, cluster candidates, train the Stage-II
cross-encoder, select settings on development data, and evaluate test timelines.
Both entry points, `run_all.sh` and `scripts/run_pipeline.py`, run this workflow.

## Repository Structure

```text
configs/                 Configurations for CREST and WCEP-CTG
dataset/                 The two datasets
pooltls/                 Core Python package
scripts/                 Command-line entry points for each stage
tests/                   Offline unit tests
requirements.txt         Python dependencies
run_all.sh               Complete workflow for one dataset
```

Local pretrained model weights belong in `models/` or another configured local
directory. Experiment outputs go to `runs/` by default. Weights, run outputs,
trained checkpoints, and caches are not included in the repository.

## Datasets and Experimental Protocol

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

`pooltls.data.DatasetReader` exposes the splits as `train`, `development`, and
`test`; `development` corresponds to the on-disk `validation/` split. Any
redistribution or use of the datasets must comply with the licenses and terms
of their original sources.

Stage-I supervision uses training articles and their gold timelines. The trained
adapter then generates Mention records independently for each split using article
text and the requested constraints. These generated mentions supply the candidate
text for clustering and timeline construction. Stage-II supervision uses training
candidates and training gold timelines. Checkpoint and fusion selection use the
development split; final metrics compare test predictions with test gold timelines.

The configured decoding protocol uses
`decoding.budget_source: reference_event_count`: the number of gold events for each
requested timeline sets its output-length budget, including on development and test
data. Gold event text supplies supervision and evaluation targets; it is not
converted into generated candidate text. Keep this budget assumption explicit when
reporting results.

## Environment and Models

Use Python 3.10 or 3.11 and a CUDA environment that supports BF16 and
bitsandbytes NF4 training. Install the pinned dependencies from the repository
root:

```bash
python -m pip install -r requirements.txt
```

The dependency file includes PyTorch, Transformers, PEFT, Accelerate,
bitsandbytes, and TILSE. Stage-I training uses 4-bit QLoRA with BF16 computation;
generation loads the Llama base model and its trained adapter in BF16. Provision
GPU memory for that full model and the configured article length and batch size.
Stage-I training requests FlashAttention 2 and falls back to PyTorch SDPA when
that backend cannot be loaded; `flash-attn` is optional with the supplied fallback.

Prepare complete local model directories with weights, configurations, and
tokenizer files. Model loading uses `local_files_only=True`; the workflow does
not download weights. The supplied configurations resolve these paths relative
to the YAML file:

| Configuration key | Default local directory | Purpose |
| --- | --- | --- |
| `paths.base_model` | `../models/Meta-Llama-3.1-8B-Instruct` | Stage-I training and article generation |
| `paths.gte_model` | `../models/gte-large` | GTE-large for semantic retrieval, clustering, supervision, negative selection, and direct scores |
| `paths.cross_encoder_model` | `../models/ms-marco-MiniLM-L-6-v2` | Stage-II cross-encoder initialization |

Place the models under the repository's `models/` directory or edit these paths
in [configs/crest.yaml](configs/crest.yaml) and
[configs/wcep_ctg.yaml](configs/wcep_ctg.yaml). Absolute paths are also accepted.
Record the actual model versions and any configuration changes with an experiment.
Stage-I training and generation preserve full article inputs and fail on inputs
that exceed their configured token limits.

Both dataset configurations use GTE-large as the frozen semantic encoder.
Stage-I supervision, candidate clustering, Stage-II supervision and negative
selection, and direct scoring on development and test data all load
`paths.gte_model`. Place the actual GTE-large weights, configuration, and tokenizer
in `models/gte-large`, or point this key to an existing local GTE-large directory.
The directory name does not change the model weights stored inside it.

## Run the Complete Workflow

Run commands from the repository root. Start each new experiment in a new or
empty run directory. A complete run trains both stages and generates events for
all three splits.

CREST, using Bash:

```bash
PYTHON_BIN=python GPU_INDEX=0 bash run_all.sh crest runs/crest_full
```

WCEP-CTG, using Bash:

```bash
PYTHON_BIN=python GPU_INDEX=0 bash run_all.sh wcep_ctg runs/wcep_ctg_full
```

The optional run-directory argument defaults to `runs/<dataset>_full` under the
repository root. `GPU_INDEX` selects the GPU exposed to the Bash launcher.

Direct Python entry point, including Windows PowerShell:

```powershell
python scripts/run_pipeline.py `
  --config configs/crest.yaml `
  --run-dir runs/crest_full `
  --device cuda:0
```

For WCEP-CTG, use `configs/wcep_ctg.yaml` and a separate run directory. The
PowerShell example requires the same CUDA and model setup as the Bash commands.

Check dataset structure, splits, and local model directories before training:

```bash
python scripts/run_pipeline.py --config configs/crest.yaml --check-only
python scripts/run_pipeline.py --config configs/wcep_ctg.yaml --check-only
```

This preflight reads the datasets and checks that configured directories exist.
It does not load model weights, test GPU compatibility, or run experiments.

## Stages and Resumption

`scripts/run_pipeline.py` executes these stages in order:

| Stage | Work performed |
| --- | --- |
| `prepare_stage1` | Align training gold events to training articles and write full-document SFT records. |
| `train_stage1` | Train the Llama QLoRA adapter and save `models/stage1/final_adapter/`. |
| `generate_train` | Generate event mentions from training articles with that adapter. |
| `generate_development` | Generate event mentions from development articles with the same adapter. |
| `generate_test` | Generate event mentions from test articles with the same adapter. |
| `cluster_all` | Cluster same-day mentions with complete linkage, independently for all splits. |
| `prepare_stage2` | Build training positives and reliable negatives screened across all constraints. |
| `train_stage2` | Train MiniLM trials and select checkpoints and fusion settings using development metrics. |
| `select_development` | Copy the selected Stage-II configuration into `selection/selected_config.json`. |
| `score_test` | Score test candidates, fuse cross-encoder and GTE-large scores, and decode with the selected settings. |
| `build_test_timelines` | Export the decoded predictions as JSONL and in the CREST timeline layout. |
| `evaluate_test` | Write TILSE ROUGE-1/ROUGE-2 and date precision, recall, and F1 metrics. |

The supplied fusion grids contain a single cross-encoder weight, `0.50`.
Development selection considers the checkpoints and settings configured for each
dataset; test data are not used to choose the checkpoint or fusion weight.

To run through Stage-I training and continue later, use a separate run directory:

```bash
python scripts/run_pipeline.py --config configs/crest.yaml --run-dir runs/crest_staged --device cuda:0 --stop-after train_stage1
python scripts/run_pipeline.py --config configs/crest.yaml --run-dir runs/crest_staged --device cuda:0 --resume --from-stage generate_train
```

For an existing run, `--resume` skips stages whose expected output files already
exist. `--from-stage` requires `--resume` and assumes that earlier stages have
finished. Resume requires a compatible `run_manifest.json` written by this
workflow. Use a fresh directory for a new experiment and keep the configuration,
datasets, and model files consistent when continuing an existing run.

After updating to PoolTLS, start in a new run directory. Run manifests now record
`method_name: PoolTLS`, and checkpoint and selection files use the `pooltls_`
identifier prefix. Earlier run directories cannot be resumed with this version.

Pipeline resumption does not automatically restore an interrupted training
optimizer. To continue Stage-I training from a saved checkpoint, replace
`checkpoint-STEP` below with an existing checkpoint directory:

```bash
python scripts/train_stage1.py --config configs/crest.yaml --train-file runs/crest_staged/stage1_data/train.jsonl --output-dir runs/crest_staged/models/stage1 --device cuda:0 --resume-from-checkpoint runs/crest_staged/models/stage1/checkpoints/checkpoint-STEP
```

Once the final adapter is saved, continue with the pipeline's `--resume` command.

## Outputs and Verification

Each run retains its configuration copy, workflow manifest, and stage logs.
Useful artifacts under the run directory include:

| Path | Contents |
| --- | --- |
| `stage1_data/` | Training records, article alignments, and supervision summary |
| `models/stage1/` | QLoRA checkpoints, final adapter, and training summary |
| `mentions/<split>/` | Article-generated event mentions and parsing metadata in `_meta/` |
| `candidates/<split>/` | Clustered candidates and clustering summary |
| `stage2_data/train.jsonl` | Stage-II supervised training pairs |
| `models/cross_encoder/` | Trial checkpoints and development selection results |
| `selection/selected_config.json` | Checkpoint and fusion settings used for test scoring |
| `scores/test/` | Cross-encoder, direct, and fused scores; decoded predictions |
| `timelines/test_predictions.jsonl` | Final test predictions |
| `timelines/crest/` | One-file-per-timeline export |
| `evaluation/test_metrics.json` | Final test evaluation metrics |
| `logs/` | Per-stage command output |

Run the offline tests with:

```bash
python -m pytest -q
```

The tests check software behavior with fixtures and mocks. Validating experiment
results requires completing training, generation, and evaluation with the actual
datasets and model weights, then inspecting the saved artifacts and metrics.
