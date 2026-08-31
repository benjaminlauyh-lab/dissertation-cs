#!/usr/bin/env python3
"""End-to-end dissertation lab experiment runner.

Trains Policy Engines (SVM / Random Forest) on human keystroke enrollment data,
optionally synthesizes VAE/GAN impersonation samples, and reports FAR/TAR.

Examples
--------
# Quick pilot (5 subjects)
python scripts/run_experiment.py --mode pilot --classifiers svm,random_forest --no-tune

# Full cohort (51 subjects)
python scripts/run_experiment.py --mode full --classifiers svm,random_forest

# Leave-one-session-out on pilot cohort
python scripts/run_experiment.py --mode pilot --split-mode losso --no-tune

# Convenience wrappers
python scripts/run_pilot.py
python scripts/run_full.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_CLASSIFIER,
    DEFAULT_ENROLL_SESSIONS,
    DEFAULT_FEATURE_SET,
    DEFAULT_IMPOSTOR_TRAIN_SAMPLES,
    DEFAULT_MODE,
    DEFAULT_SPLIT_MODE,
    DEFAULT_SYNTHETIC_SAMPLES,
    DEFAULT_VERIFY_SESSIONS,
    FIGURES_DIR,
    GAN_DEFAULTS,
    RANDOM_SEED,
    RESULTS_DIR,
    VAE_DEFAULTS,
    set_global_seed,
)
from src.data.loader import load_dataset  # noqa: E402
from src.data.splits import build_leave_one_session_out_splits  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    AttackMetrics,
    MetricsResult,
    build_hypothesis_tests,
    build_statistical_summary,
    summarize_attack_results,
)
from src.experiment.runner import (  # noqa: E402
    ensure_results_dirs,
    make_run_id,
    resolve_subjects,
    run_single_subject,
    write_run_meta,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UEBA resilience experiment.")
    parser.add_argument(
        "--mode",
        choices=("pilot", "full"),
        default=DEFAULT_MODE,
        help="pilot = first 5 subjects; full = all 51 (default: pilot).",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Override mode: number, comma-separated IDs, or 'all'.",
    )
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        help=f"Feature set (default: {DEFAULT_FEATURE_SET}).",
    )
    parser.add_argument(
        "--classifiers",
        default=DEFAULT_CLASSIFIER,
        help="Comma-separated classifiers: svm,random_forest",
    )
    parser.add_argument(
        "--attacks",
        default="vae,gan",
        help="Comma-separated attack methods: vae,gan",
    )
    parser.add_argument(
        "--skip-attacks",
        action="store_true",
        help="Skip generative synthesis; report human-only baseline.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("fixed", "losso"),
        default=DEFAULT_SPLIT_MODE,
        help="fixed = enroll 1–4 / verify 5; losso = leave-one-session-out.",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=DEFAULT_SYNTHETIC_SAMPLES,
        help="Synthetic samples per user per attack method.",
    )
    parser.add_argument(
        "--impostor-samples",
        type=int,
        default=DEFAULT_IMPOSTOR_TRAIN_SAMPLES,
        help="Max human impostor samples for PE training.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable GridSearchCV hyperparameter tuning (faster).",
    )
    parser.add_argument(
        "--vae-epochs",
        type=int,
        default=VAE_DEFAULTS["epochs"],
    )
    parser.add_argument(
        "--gan-epochs",
        type=int,
        default=GAN_DEFAULTS["epochs"],
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (cpu or cuda/mps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Skip saving PE / VAE / GAN model files (faster I/O).",
    )
    return parser.parse_args(argv)


def plot_far_comparison(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    has_ci = "far_human_ci_low" in summary.columns

    rows = []
    for _, row in summary.iterrows():
        label = f"{str(row['method']).upper()} / {row['classifier_type']}"
        for far_type, mean_col, lo_col, hi_col in (
            ("Human impostor FAR", "far_human_mean", "far_human_ci_low", "far_human_ci_high"),
            ("Synthetic attack FAR", "far_attack_mean", "far_attack_ci_low", "far_attack_ci_high"),
        ):
            mean = float(row[mean_col])
            if has_ci:
                err_lo = max(0.0, mean - float(row[lo_col]))
                err_hi = max(0.0, float(row[hi_col]) - mean)
            else:
                err_lo = err_hi = 0.0
            rows.append(
                {
                    "label": label,
                    "far_type": far_type,
                    "far": mean,
                    "err_lo": err_lo,
                    "err_hi": err_hi,
                }
            )
    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=plot_df, x="label", y="far", hue="far_type")
    if has_ci:
        # seaborn draws hue groups; match patches to plot_df row order
        for patch, (_, prow) in zip(ax.patches, plot_df.iterrows()):
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            ax.errorbar(
                x,
                y,
                yerr=[[prow["err_lo"]], [prow["err_hi"]]],
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1,
            )
    plt.ylabel("False Acceptance Rate")
    plt.xlabel("")
    plt.title("UEBA FAR: Human Impostors vs Generative Impersonation (mean ± 95% CI)")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_human_baseline(rows: list[MetricsResult], path: Path) -> None:
    if not rows:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.__dict__ for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(
        axes,
        ["tar", "far", "eer"],
        ["TAR", "FAR (human impostors)", "EER"],
    ):
        sns.boxplot(data=df, y=col, ax=ax, color="teal")
        ax.set_title(title)
        ax.set_ylim(0, 1)
    fig.suptitle("Human-only Policy Engine baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _print_hypothesis_block(hyp: pd.DataFrame) -> None:
    if hyp.empty:
        return
    print("\n--- Hypothesis tests (attack FAR > human FAR) ---")
    for _, row in hyp.iterrows():
        p = row["p_value"]
        p_str = f"{p:.4g}" if pd.notna(p) else "n/a"
        print(
            f"  {str(row['method']).upper()} / {row['classifier_type']}: "
            f"ΔFAR={row['mean_delta']:.4f} ({row['test_name']} p={p_str})"
        )
    # Short console claim using the first available Wilcoxon/t result
    first = hyp.iloc[0]
    p = first["p_value"]
    p_str = f"{p:.4g}" if pd.notna(p) else "n/a"
    print(
        f"Attack FAR > human FAR ({first['test_name']} p={p_str}) "
        f"[{first['method']}/{first['classifier_type']}; see hypothesis_tests.csv]"
    )


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)
    ensure_results_dirs()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    subjects_explicit = args.subjects is not None
    print("Loading CMU keystroke dataset...")
    df = load_dataset()
    subjects = resolve_subjects(
        df,
        mode=args.mode,
        subjects_arg=args.subjects if subjects_explicit else None,
        subjects_explicit=subjects_explicit,
    )
    # If --subjects all under pilot mode override → treat resolved as full for tagging
    mode = args.mode
    if subjects_explicit and args.subjects is not None:
        all_n = len(resolve_subjects(df, mode="full", subjects_explicit=False))
        if len(subjects) >= all_n:
            mode = "full"
        elif args.subjects.strip().lower() != "all" and "," not in args.subjects:
            # numeric override keeps declared mode label unless it's clearly full
            pass

    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    attacks = (
        []
        if args.skip_attacks
        else [a.strip() for a in args.attacks.split(",") if a.strip()]
    )
    run_id = make_run_id(mode, args.seed)
    save_artifacts = not args.no_save_artifacts

    print(
        f"mode={mode} | run_id={run_id} | Subjects={len(subjects)} | "
        f"feature_set={args.feature_set} | split_mode={args.split_mode} | "
        f"classifiers={classifiers} | attacks={attacks or ['none']}"
    )

    baseline_rows: list[MetricsResult] = []
    attack_rows: list[AttackMetrics] = []
    baseline_records: list[dict] = []
    attack_records: list[dict] = []
    cross_session_records: list[dict] = []

    for subject in subjects:
        print(f"\n=== Subject {subject} ===")
        if args.split_mode == "losso":
            for fold in build_leave_one_session_out_splits(
                df, subject, feature_set=args.feature_set
            ):
                v = fold.verify_session
                print(f"  -- LOSO fold verify_session={v} --")
                result = run_single_subject(
                    df,
                    subject,
                    feature_set=args.feature_set,
                    classifiers=classifiers,
                    attacks=attacks,
                    split=fold,
                    impostor_samples=args.impostor_samples,
                    n_synthetic=args.n_synthetic,
                    tune_hyperparameters=not args.no_tune,
                    vae_epochs=args.vae_epochs,
                    gan_epochs=args.gan_epochs,
                    device=args.device,
                    seed=args.seed,
                    save_artifacts=save_artifacts,
                    mode=mode,
                    run_id=run_id,
                    split_mode="losso",
                )
                baseline_rows.extend(result.baseline_metrics)
                attack_rows.extend(result.attack_rows)
                baseline_records.extend(result.baseline_records)
                attack_records.extend(result.attack_records)
                for rec in result.baseline_records:
                    cross_session_records.append({**rec, "record_type": "baseline"})
                for rec in result.attack_records:
                    cross_session_records.append({**rec, "record_type": "attack"})
        else:
            result = run_single_subject(
                df,
                subject,
                feature_set=args.feature_set,
                classifiers=classifiers,
                attacks=attacks,
                enroll_sessions=DEFAULT_ENROLL_SESSIONS,
                verify_sessions=DEFAULT_VERIFY_SESSIONS,
                impostor_samples=args.impostor_samples,
                n_synthetic=args.n_synthetic,
                tune_hyperparameters=not args.no_tune,
                vae_epochs=args.vae_epochs,
                gan_epochs=args.gan_epochs,
                device=args.device,
                seed=args.seed,
                save_artifacts=save_artifacts,
                mode=mode,
                run_id=run_id,
                split_mode="fixed",
            )
            baseline_rows.extend(result.baseline_metrics)
            attack_rows.extend(result.attack_rows)
            baseline_records.extend(result.baseline_records)
            attack_records.extend(result.attack_records)

    # Persist tables
    baseline_df = pd.DataFrame(baseline_records)
    baseline_path = RESULTS_DIR / "human_baseline.csv"
    baseline_df.to_csv(baseline_path, index=False)
    print(f"\nSaved human baseline → {baseline_path}")

    summary = summarize_attack_results(attack_rows)
    if not summary.empty:
        attack_detail = pd.DataFrame(attack_records)
        detail_path = RESULTS_DIR / "attack_results.csv"
        summary_path = RESULTS_DIR / "attack_summary.csv"
        attack_detail.to_csv(detail_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"Saved attack detail → {detail_path}")
        print(f"Saved attack summary → {summary_path}")
        print("\nAttack summary:")
        print(summary.to_string(index=False))

        stats_df = build_statistical_summary(attack_rows)
        stats_path = RESULTS_DIR / "statistical_summary.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved statistical summary → {stats_path}")

        hyp_df = build_hypothesis_tests(attack_rows)
        hyp_path = RESULTS_DIR / "hypothesis_tests.csv"
        hyp_df.to_csv(hyp_path, index=False)
        print(f"Saved hypothesis tests → {hyp_path}")
        _print_hypothesis_block(hyp_df)

        plot_far_comparison(summary, FIGURES_DIR / "far_comparison.png")
        print(f"Saved FAR comparison figure → {FIGURES_DIR / 'far_comparison.png'}")

    if args.split_mode == "losso" and cross_session_records:
        cross_df = pd.DataFrame(cross_session_records)
        cross_path = RESULTS_DIR / "cross_session_results.csv"
        cross_df.to_csv(cross_path, index=False)
        print(f"Saved cross-session results → {cross_path}")
        # Aggregate mean±std across folds and subjects for console
        if attack_rows:
            print("\nCross-session attack FAR (mean ± std over folds×subjects):")
            agg = (
                pd.DataFrame([r.__dict__ for r in attack_rows])
                .groupby(["method", "classifier_type"], as_index=False)
                .agg(
                    far_attack_mean=("far_attack", "mean"),
                    far_attack_std=("far_attack", "std"),
                    far_human_mean=("far_human", "mean"),
                    far_human_std=("far_human", "std"),
                )
                .fillna(0.0)
            )
            print(agg.to_string(index=False))

    plot_human_baseline(baseline_rows, FIGURES_DIR / "human_baseline.png")
    print(f"Saved human baseline figure → {FIGURES_DIR / 'human_baseline.png'}")

    write_run_meta(
        RESULTS_DIR / "run_meta.json",
        mode=mode,
        run_id=run_id,
        seed=args.seed,
        subjects=subjects,
        feature_set=args.feature_set,
        classifiers=classifiers,
        attacks=attacks,
        split_mode=args.split_mode,
        enroll_sessions=DEFAULT_ENROLL_SESSIONS,
        verify_sessions=DEFAULT_VERIFY_SESSIONS,
        n_synthetic=args.n_synthetic,
        vae_epochs=args.vae_epochs,
        gan_epochs=args.gan_epochs,
        extra={
            "tune_hyperparameters": not args.no_tune,
            "device": args.device,
            "impostor_samples": args.impostor_samples,
        },
    )
    print(f"Saved run metadata → {RESULTS_DIR / 'run_meta.json'}")
    print("Done.")


if __name__ == "__main__":
    run()
