# ResNet20 CIFAR-100 Compression: Pruning + Knowledge Distillation + QAT

A complete, reproducible pipeline for compressing a ResNet20 classifier on CIFAR-100
using three complementary techniques:

- **Iterative structural filter pruning** (L1-norm filter ranking), compared **with**
  and **without** knowledge distillation during fine-tuning (Scenarios A/B).
- **Ensemble knowledge distillation (KD)** from a 3-model teacher ensemble
  (ResNet32, ResNet56, WideResNet-28-10).
- **Quantization-aware training (QAT)** at INT8 and INT4, again compared **with** and
  **without** KD (Scenarios C/D).

This repository is a refactor of an exploratory Jupyter notebook into a tested,
importable Python package with a single command-line entry point. See
[`NOTES.md`](NOTES.md) for two correctness bugs that were found in the original
notebook and fixed here, and for other behavioral notes worth knowing before you rely
on the numbers.

## Table of contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [What the pipeline does](#what-the-pipeline-does)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Resumability](#resumability)
- [Reading the results](#reading-the-results)
- [Testing](#testing)
- [Notebook](#notebook)
- [License](#license)

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -e .
```

This pulls in `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib` and `tqdm` (see
[`pyproject.toml`](pyproject.toml)).

> **Tip:** a plain `pip install torch` fetches the CUDA build, which drags in several
> GB of NVIDIA wheels even on a CPU-only machine or a Colab instance that already has
> a GPU build of torch preinstalled. If you hit disk-space limits or already have torch
> installed, skip installing it from this project's dependencies and use the selector
> at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to get
> the right command for your platform first, then run `pip install -e . --no-deps`.

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

## Quickstart

Run the full pipeline with default settings (CIFAR-100 downloads automatically to
`./data`):

```bash
resnet-compress
```

or equivalently:

```bash
python -m resnet_compression
```

Verify the installation instantly, without downloading data or doing any real training
(a few seconds, on CPU, with synthetic data):

```bash
resnet-compress --smoke-test
```

Common options:

```bash
resnet-compress \
  --output-dir ./filter_pruning_experiment \
  --pruning-rates 0.02 0.05 0.10 \
  --initial-epochs 200 \
  --finetune-epochs 20 \
  --qat-epochs 20 \
  --acceptable-acc-drop 5.0 \
  --device auto
```

Run `resnet-compress --help` for the full list.

The default schedule (10 pruning rates x up to 100 prune/fine-tune iterations each,
200-epoch baseline and teacher training) is the same one used in the original
experiment and is **not** fast: expect it to take on the order of a day on a single
modern GPU. Use `--pruning-rates`, `--initial-epochs`, `--finetune-epochs`,
`--prune-iterations` and `--qat-epochs` to scale it down.

## What the pipeline does

The stages run in this order (mirroring [`resnet_compression/cli.py`](resnet_compression/cli.py)):

1. **Baseline**: train a plain ResNet20 on CIFAR-100.
2. **Teachers**: train a ResNet32, ResNet56 and WideResNet-28-10 ensemble.
3. **Scenario A**: for every target pruning rate, repeatedly prune the lowest-L1-norm
   filters of every residual block and fine-tune **with** ensemble KD, stopping when
   the accuracy drop exceeds `--acceptable-acc-drop`, pruning stalls out, or
   `--prune-iterations` is reached.
4. **Scenario B**: the same loop, fine-tuning with plain cross-entropy instead of KD.
5. **Selection**: pick the pruned model (across every rate, in both scenarios) with the
   highest compression ratio among those meeting the accuracy-drop threshold.
6. **Scenario C**: quantization-aware training of the selected model at INT8, then
   independently at INT4, fine-tuning **with** KD.
7. **Scenario D**: the same, fine-tuning **without** KD.
8. **Reporting**: three comparison plots, a CSV summary table, and a text report with
   headline numbers and recommendations.

See the module docstrings in [`resnet_compression/pruning.py`](resnet_compression/pruning.py)
and [`resnet_compression/quantization.py`](resnet_compression/quantization.py) for the
exact algorithms.

## Configuration

All settings live in [`resnet_compression/config.py`](resnet_compression/config.py)'s
`Config` dataclass and are exposed as CLI flags. The most relevant:

| Flag | Default | Meaning |
|---|---|---|
| `--pruning-rates` | `0.01 ... 0.10` (step `0.01`) | Target rates tried in Scenarios A/B |
| `--acceptable-acc-drop` | `5.0` | Max accuracy drop (%) before a scenario stops pruning further |
| `--initial-epochs` | `200` | Epochs for the baseline and each teacher |
| `--finetune-epochs` | `20` | Epochs per prune/fine-tune iteration |
| `--qat-epochs` | `20` | Epochs per QAT bit-width |
| `--prune-iterations` | `100` | Max prune/fine-tune iterations per rate |
| `--qat-backend` | `fbgemm` | INT8 backend qconfig (`fbgemm`, `x86`, `onednn`, `qnnpack`; use `qnnpack` on ARM, e.g. Apple Silicon or Raspberry Pi) |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, ... |
| `--force` | off | Recompute every stage even if cached results exist |
| `--smoke-test` | off | Tiny synthetic run to check the installation |

## Project structure

```
.
├── resnet_compression/          # the installable package
│   ├── config.py                 # Config dataclass (all hyperparameters)
│   ├── data.py                   # CIFAR-100 loading + transforms
│   ├── models.py                 # ResNet20/32/56, WideResNet, pruned-model rebuilding
│   ├── training.py               # train/eval loops, ensemble KD loss
│   ├── pruning.py                # L1 structural filter pruning
│   ├── quantization.py           # QAT wrapper, INT8/INT4 qconfigs, conversion
│   ├── pruning_pipeline.py       # Scenario A/B orchestration
│   ├── qat_pipeline.py           # Scenario C/D orchestration
│   ├── selection.py              # best-model selection between scenarios
│   ├── visualization.py          # the 3 result plots
│   ├── reporting.py              # summary table + text report
│   ├── utils.py                  # logging, checkpoint/JSON I/O, model-size helpers
│   └── cli.py                    # `resnet-compress` entry point
├── tests/                        # pytest suite (fast, CPU-only, synthetic data)
├── notebooks/
│   └── demo.ipynb                # short Colab-friendly walkthrough
├── pyproject.toml
├── requirements.txt
├── NOTES.md                      # bugs found & fixed vs. the original notebook
└── README.md
```

Running the pipeline creates an output directory (default `./filter_pruning_experiment/`,
override with `--output-dir`) with this layout:

```
filter_pruning_experiment/
├── checkpoints/                  # baseline_resnet20.pth, teacher_*.pth
├── scenario_a_with_kd/           # per-rate checkpoints, final_model_*.pth, scenario_a_results.json
├── scenario_b_no_kd/             # same, for Scenario B
├── scenario_c_qat_with_kd/       # best_model_int8_with_kd.pth, best_model_int4_with_kd.pth, scenario_c_results.json
├── scenario_d_qat_no_kd/         # same, for Scenario D
├── plots/                        # plot1_pruning_comparison.png, plot2_pareto_fronts.png, plot3_qat_comparison.png
├── baseline_metrics.json
├── teacher_metrics.json
├── best_model_for_qat.json
├── complete_results_summary.csv
└── pipeline.log
```

This directory is git-ignored by default (see [`.gitignore`](.gitignore)): it is
regenerated by re-running the pipeline and typically contains hundreds of MB of
checkpoints, which don't belong in version control. If you want to publish your own
run's results, commit the small JSON/CSV/PNG files explicitly rather than removing them
from `.gitignore` wholesale.

## Resumability

Every stage checks whether its output already exists (a checkpoint, a
`scenario_*_results.json`, `best_model_for_qat.json`) and loads it instead of
retraining, unless `--force` is passed. Re-running the same command after an
interruption (a Colab timeout, a crashed job) continues roughly where it left off. To
force a clean re-run of everything, pass `--force` or point `--output-dir` at a new,
empty directory.

## Reading the results

`complete_results_summary.csv` (and the same table printed at the end of the run) has
one row per produced model: the FP32 baseline, every pruning rate of Scenarios A and B,
and both bit-widths of Scenarios C and D, each with accuracy, accuracy drop from
baseline, parameter count, size in MB, and compression ratio. The three plots in
`plots/` visualize, respectively: accuracy/compression/size across pruning rates
(A vs. B); Pareto fronts of accuracy vs. compression, including the full pipeline's
path from baseline to INT4; and a KD-vs-no-KD comparison at each QAT bit-width.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite (`tests/`) runs on CPU with tiny synthetic models and data — it checks
architecture shapes/parameter counts, pruning invariants (including a regression test
for the INT8/INT4 bug described in `NOTES.md`), config validation, and utility
functions. It does not reproduce the full multi-hour experiment; use `--smoke-test` for
an end-to-end (but numerically meaningless) sanity check of the real pipeline.

## Notebook

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) is a short, Colab-friendly walkthrough:
it installs the package, builds and prunes a model interactively, and runs a
`--smoke-test`-scale pipeline end to end so you can see the plots and summary table
without waiting hours. For an actual full experiment, use the `resnet-compress` CLI
(optionally from a notebook cell via `!resnet-compress ...`) rather than a notebook —
long-running training is easier to resume, log and monitor from the command line.

## License

[MIT](LICENSE).
