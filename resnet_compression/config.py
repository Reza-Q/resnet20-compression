"""Experiment configuration.

The defaults reproduce the experiment described in the README. Every value can be
overridden from the command line (see ``resnet_compression.cli``) or by creating a
:class:`Config` instance directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Sub-directory (inside ``output_dir``) used by each experimental scenario.
SCENARIO_DIRS: dict[str, str] = {
    "A": "scenario_a_with_kd",  # iterative pruning + knowledge distillation
    "B": "scenario_b_no_kd",  # iterative pruning + plain fine-tuning
    "C": "scenario_c_qat_with_kd",  # quantization-aware training + KD
    "D": "scenario_d_qat_no_kd",  # quantization-aware training, no KD
}

SUPPORTED_QAT_BACKENDS = ("fbgemm", "x86", "onednn", "qnnpack")

#: Number of CIFAR-100 classes.
NUM_CLASSES = 100


def default_pruning_rates() -> list[float]:
    """The 1%-10% (step 1%) schedule used unless ``--pruning-rates`` overrides it.

    A plain function rather than a class attribute, because a dataclass field declared
    with ``field(default_factory=...)`` (needed here to avoid every :class:`Config`
    instance sharing one mutable list) has no corresponding value on the class itself:
    ``Config.pruning_rates`` raises ``AttributeError`` unless you go through an instance.
    """
    return [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]


@dataclass
class Config:
    """All tunable settings of the pipeline."""

    # --- Schedule -----------------------------------------------------------------
    initial_epochs: int = 200  # epochs used to train the baseline and every teacher
    prune_iterations: int = 100  # maximum number of prune + fine-tune iterations per rate
    finetune_epochs: int = 20  # fine-tuning epochs after every pruning iteration
    qat_epochs: int = 20  # QAT epochs per bit-width

    # --- Pruning ------------------------------------------------------------------
    pruning_rates: list[float] = field(default_factory=default_pruning_rates)
    acceptable_acc_drop: float = 5.0  # stop when accuracy falls this far below baseline (%)
    checkpoint_interval: float = 0.02  # save an intermediate model every 2% of pruned params

    # --- Optimisation -------------------------------------------------------------
    batch_size: int = 128
    test_batch_size: int = 100
    initial_lr: float = 0.1
    finetune_lr: float = 0.001
    qat_lr: float = 1e-4
    kd_temperature: float = 4.0
    kd_alpha: float = 0.7  # weight of the distillation loss (1 - alpha: hard-label loss)

    # --- Quantization -------------------------------------------------------------
    qat_backend: str = "fbgemm"  # use "qnnpack" on ARM CPUs (e.g. Apple silicon)

    # --- Runtime ------------------------------------------------------------------
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:1", ...
    num_workers: int = 4
    seed: int = 42
    data_dir: str = "./data"
    output_dir: str = "./filter_pruning_experiment"
    smoke_test: bool = False  # tiny synthetic run used to check the installation
    force: bool = False  # recompute selected stages even if results already exist

    # ------------------------------------------------------------------ paths ------
    @property
    def base_dir(self) -> Path:
        return Path(self.output_dir)

    @property
    def checkpoint_dir(self) -> Path:
        return self.base_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        return self.base_dir / "plots"

    def scenario_dir(self, scenario: str) -> Path:
        """Directory holding the models and results of scenario ``A``, ``B``, ``C`` or ``D``."""
        return self.base_dir / SCENARIO_DIRS[scenario.upper()]

    def results_path(self, scenario: str) -> Path:
        """JSON file with the results of a scenario."""
        return self.scenario_dir(scenario) / f"scenario_{scenario.lower()}_results.json"

    def make_dirs(self) -> None:
        """Create the output directory tree."""
        for path in [self.base_dir, self.checkpoint_dir, self.plots_dir]:
            path.mkdir(parents=True, exist_ok=True)
        for scenario in SCENARIO_DIRS:
            self.scenario_dir(scenario).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- helpers ------
    def apply_smoke_test(self) -> Config:
        """Shrink the experiment so that the whole pipeline finishes within minutes.

        Uses synthetic data and tiny teachers, so the resulting accuracies are
        meaningless; the run only verifies that every stage executes.
        """
        self.smoke_test = True
        self.initial_epochs = 1
        self.finetune_epochs = 1
        self.qat_epochs = 1
        self.prune_iterations = 2
        self.pruning_rates = [0.05, 0.10]
        self.batch_size = 64
        self.test_batch_size = 64
        self.num_workers = 0
        return self

    def validate(self) -> None:
        """Raise ``ValueError`` if a setting is out of range."""
        if not self.pruning_rates:
            raise ValueError("pruning_rates must contain at least one value")
        for rate in self.pruning_rates:
            if not 0.0 < rate < 1.0:
                raise ValueError(f"pruning rates must lie in (0, 1); got {rate}")
        for name in ("initial_epochs", "finetune_epochs", "qat_epochs", "prune_iterations"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.batch_size < 1 or self.test_batch_size < 1:
            raise ValueError("batch sizes must be >= 1")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0")
        if not 0.0 <= self.kd_alpha <= 1.0:
            raise ValueError("kd_alpha must lie in [0, 1]")
        if self.qat_backend not in SUPPORTED_QAT_BACKENDS:
            raise ValueError(f"qat_backend must be one of {SUPPORTED_QAT_BACKENDS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
