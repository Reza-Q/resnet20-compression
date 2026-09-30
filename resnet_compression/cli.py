"""End-to-end pipeline: baseline -> teachers -> pruning (A/B) -> selection -> QAT (C/D) -> report.

Every stage is resumable: it is skipped and loaded from disk if its output already
exists, unless ``--force`` is given. Re-running the same command after an interruption
therefore continues roughly where it left off, exactly like the original notebook's
"checkpoint or train" cells.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import Config, default_pruning_rates
from .data import get_dataloaders
from .models import SMOKE_TEACHER_SPECS, TEACHER_SPECS, resnet20
from .pruning_pipeline import run_pruning_scenario
from .qat_pipeline import run_qat_scenario
from .reporting import build_summary_table, compute_insights, format_final_report, save_summary_csv
from .selection import load_pruned_model_for_qat, select_best_model
from .training import train_model
from .utils import (
    count_conv_filters,
    count_parameters,
    get_device,
    get_model_size_mb,
    load_checkpoint,
    log_section,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logging,
)
from .visualization import plot_pareto_fronts, plot_pruning_comparison, plot_qat_comparison

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet20/CIFAR-100 iterative pruning, knowledge distillation and QAT pipeline."
    )
    parser.add_argument("--output-dir", default=Config.output_dir, help="Root directory for all outputs.")
    parser.add_argument("--data-dir", default=Config.data_dir, help="Where CIFAR-100 is stored/downloaded.")
    parser.add_argument("--device", default="auto", help='"auto", "cpu", "cuda", "cuda:0", ...')
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--initial-epochs", type=int, default=Config.initial_epochs, help="Baseline + teacher epochs.")
    parser.add_argument("--finetune-epochs", type=int, default=Config.finetune_epochs, help="Epochs per pruning iteration.")
    parser.add_argument("--qat-epochs", type=int, default=Config.qat_epochs)
    parser.add_argument("--prune-iterations", type=int, default=Config.prune_iterations, help="Max iterations per pruning rate.")
    parser.add_argument(
        "--pruning-rates",
        type=float,
        nargs="+",
        default=None,
        help="Target rates as fractions, e.g. --pruning-rates 0.01 0.02 0.05. Defaults to 1%%-10%% in 1%% steps"
        f" ({[f'{r * 100:.0f}%' for r in default_pruning_rates()]}).",
    )
    parser.add_argument("--acceptable-acc-drop", type=float, default=Config.acceptable_acc_drop)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--qat-backend", default=Config.qat_backend, choices=["fbgemm", "x86", "onednn", "qnnpack"])
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--force", action="store_true", help="Recompute every stage even if results already exist.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny synthetic run (1-2 epochs, random data, no download) to verify the pipeline runs end to end.",
    )
    parser.add_argument("--log-file", default=None, help="Optional path to also write logs to a file.")
    args = parser.parse_args(argv)
    return args


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        device=args.device,
        seed=args.seed,
        initial_epochs=args.initial_epochs,
        finetune_epochs=args.finetune_epochs,
        qat_epochs=args.qat_epochs,
        prune_iterations=args.prune_iterations,
        acceptable_acc_drop=args.acceptable_acc_drop,
        batch_size=args.batch_size,
        qat_backend=args.qat_backend,
        num_workers=args.num_workers,
        force=args.force,
    )
    if args.pruning_rates:
        cfg.pruning_rates = args.pruning_rates
    if args.smoke_test:
        cfg.apply_smoke_test()
    cfg.validate()
    return cfg


def run(cfg: Config) -> int:
    cfg.make_dirs()
    setup_logging(log_file=cfg.base_dir / "pipeline.log")
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    log_section("ResNet20 / CIFAR-100 adaptive filter pruning pipeline")
    logger.info("Device: %s", device)
    logger.info("Output directory: %s", cfg.base_dir)
    logger.info("Pruning rates: %s", [f"{r * 100:.0f}%" for r in cfg.pruning_rates])
    if cfg.smoke_test:
        logger.warning("Smoke-test mode: synthetic data, tiny epoch counts. Accuracies are meaningless.")

    train_loader, test_loader = get_dataloaders(cfg)

    # ---- Baseline ---------------------------------------------------------------
    log_section("Baseline ResNet20")
    baseline_checkpoint = cfg.checkpoint_dir / "baseline_resnet20.pth"
    baseline_info, loaded = (None, False) if cfg.force else load_checkpoint(baseline_checkpoint, device)
    if loaded:
        baseline_model = resnet20()
        baseline_model.load_state_dict(baseline_info["model_state"])
        baseline_acc = baseline_info["accuracy"]
    else:
        baseline_model, baseline_acc, history = train_model(
            resnet20(), train_loader, test_loader, epochs=cfg.initial_epochs, lr=cfg.initial_lr,
            device=device, model_name="Baseline ResNet20",
        )
        save_checkpoint(
            {"model_state": baseline_model.state_dict(), "accuracy": baseline_acc, "history": history},
            baseline_checkpoint,
        )

    baseline_model = baseline_model.cpu()
    baseline_params = count_parameters(baseline_model)
    baseline_size_mb = get_model_size_mb(baseline_model)
    logger.info(
        "Baseline: accuracy=%.2f%%, params=%s, filters=%d, size=%.3f MB",
        baseline_acc, f"{baseline_params:,}", count_conv_filters(baseline_model), baseline_size_mb,
    )
    save_json(
        {
            "accuracy": baseline_acc,
            "parameters": baseline_params,
            "filters": count_conv_filters(baseline_model),
            "size_mb": baseline_size_mb,
        },
        cfg.base_dir / "baseline_metrics.json",
    )

    # ---- Teacher ensemble ---------------------------------------------------------
    log_section("Teacher ensemble")
    teacher_specs = SMOKE_TEACHER_SPECS if cfg.smoke_test else TEACHER_SPECS
    teachers = []
    teacher_metrics = {}
    for checkpoint_file, model_fn, name in teacher_specs:
        path = cfg.checkpoint_dir / checkpoint_file
        teacher_info, loaded = (None, False) if cfg.force else load_checkpoint(path, device)
        if loaded:
            teacher_model = model_fn()
            teacher_model.load_state_dict(teacher_info["model_state"])
            teacher_acc = teacher_info["accuracy"]
        else:
            teacher_model, teacher_acc, _ = train_model(
                model_fn(), train_loader, test_loader, epochs=cfg.initial_epochs, lr=cfg.initial_lr,
                device=device, model_name=name,
            )
            save_checkpoint({"model_state": teacher_model.state_dict(), "accuracy": teacher_acc}, path)
        teachers.append(teacher_model)
        teacher_metrics[name] = teacher_acc
        logger.info("%s: %.2f%%", name, teacher_acc)

    save_json(
        {"teachers": teacher_metrics, "ensemble_avg": sum(teacher_metrics.values()) / len(teacher_metrics)},
        cfg.base_dir / "teacher_metrics.json",
    )

    # ---- Scenarios A & B: iterative pruning ----------------------------------------
    log_section("Scenario A: iterative pruning WITH knowledge distillation")
    results_a = run_pruning_scenario(
        cfg, "A", use_kd=True, baseline_model=baseline_model, baseline_acc=baseline_acc,
        baseline_params=baseline_params, teachers=teachers, train_loader=train_loader,
        test_loader=test_loader, device=device,
    )

    log_section("Scenario B: iterative pruning WITHOUT knowledge distillation")
    results_b = run_pruning_scenario(
        cfg, "B", use_kd=False, baseline_model=baseline_model, baseline_acc=baseline_acc,
        baseline_params=baseline_params, teachers=teachers, train_loader=train_loader,
        test_loader=test_loader, device=device,
    )

    # ---- Selection ------------------------------------------------------------------
    log_section("Selecting the best pruned model for QAT")
    best_info = select_best_model(cfg, results_a, results_b, baseline_acc)
    pruned_model = load_pruned_model_for_qat(cfg, best_info)

    # ---- Scenarios C & D: quantization-aware training --------------------------------
    log_section("Scenario C: QAT WITH knowledge distillation (INT8 & INT4)")
    results_c = run_qat_scenario(
        cfg, "C", use_kd=True, pruned_model=pruned_model, base_info=best_info.to_dict(),
        baseline_acc=baseline_acc, baseline_size_mb=baseline_size_mb, teachers=teachers,
        train_loader=train_loader, test_loader=test_loader, device=device,
    )

    log_section("Scenario D: QAT WITHOUT knowledge distillation (INT8 & INT4)")
    results_d = run_qat_scenario(
        cfg, "D", use_kd=False, pruned_model=pruned_model, base_info=best_info.to_dict(),
        baseline_acc=baseline_acc, baseline_size_mb=baseline_size_mb, teachers=teachers,
        train_loader=train_loader, test_loader=test_loader, device=device,
    )

    # ---- Plots & final report ---------------------------------------------------------
    log_section("Plots and final report")
    plot_pruning_comparison(cfg, results_a, results_b, baseline_acc, baseline_size_mb)
    plot_pareto_fronts(cfg, results_a, results_b, results_c, baseline_acc)
    plot_qat_comparison(cfg, results_c, results_d)

    summary = build_summary_table(cfg, baseline_acc, baseline_params, baseline_size_mb, results_a, results_b, results_c, results_d)
    save_summary_csv(cfg, summary)
    insights = compute_insights(cfg, results_a, results_b, results_c, results_d)
    report = format_final_report(cfg, summary, insights, results_a, results_b, results_c, results_d)
    logger.info("\n%s", report)

    log_section("Pipeline finished")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    start = time.time()
    try:
        return run(cfg)
    finally:
        logger.info("Total wall-clock time: %.1f minutes", (time.time() - start) / 60)


if __name__ == "__main__":
    sys.exit(main())
