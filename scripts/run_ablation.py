#!/usr/bin/env python3
"""Ablation experiments: enrollment size and feature-set sensitivity.

Default scope (fast proof): 5 subjects, SVM only, VAE only, --no-tune,
reduced epochs. Expand via flags for a full matrix later.

Outputs
-------
results/ablation_enrollment.csv
results/ablation_features.csv
results/figures/ablation_enrollment.png
results/figures/ablation_features.png
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
    DEFAULT_FEATURE_SET,
    DEFAULT_IMPOSTOR_TRAIN_SAMPLES,
    DEFAULT_SYNTHETIC_SAMPLES,
    DEFAULT_VERIFY_SESSIONS,
    FEATURE_SETS,
    FIGURES_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    set_global_seed,
)
from src.data.loader import load_dataset  # noqa: E402
from src.experiment.runner import (  # noqa: E402
    ensure_results_dirs,
    make_run_id,
    resolve_subjects,
    run_single_subject,
    write_run_meta,
)

ENROLLMENT_SCHEDULE = (
    (1,),
    (1, 2),
    (1, 2, 3),
    (1, 2, 3, 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run enrollment / feature ablations.")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--subjects", default=None, help="Override mode subject selection.")
    parser.add_argument(
        "--classifiers",
        default="svm",
        help="Comma-separated classifiers (default: svm).",
    )
    parser.add_argument(
        "--attacks",
        default="vae",
        help="Comma-separated attacks (default: vae).",
    )
    parser.add_argument(
        "--feature-sets",
        default=",".join(FEATURE_SETS.keys()),
        help="Comma-separated feature sets for feature ablation.",
    )
    parser.add_argument(
        "--skip-enrollment",
        action="store_true",
        help="Skip enrollment-size ablation.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature-set ablation.",
    )
    parser.add_argument("--n-synthetic", type=int, default=DEFAULT_SYNTHETIC_SAMPLES)
    parser.add_argument("--impostor-samples", type=int, default=DEFAULT_IMPOSTOR_TRAIN_SAMPLES)
    parser.add_argument("--no-tune", action="store_true", default=True)
    parser.add_argument("--tune", action="store_true", help="Enable GridSearchCV.")
    parser.add_argument("--vae-epochs", type=int, default=80)
    parser.add_argument("--gan-epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def _plot_enrollment(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    agg = (
        df.groupby(["n_enroll_sessions", "method", "classifier_type"], as_index=False)
        .agg(far_attack=("far_attack", "mean"), far_human=("far_human", "mean"))
    )
    plt.figure(figsize=(8, 4.5))
    sns.lineplot(
        data=agg,
        x="n_enroll_sessions",
        y="far_attack",
        hue="method",
        style="classifier_type",
        marker="o",
    )
    plt.ylabel("Attack FAR (mean)")
    plt.xlabel("Number of enrollment sessions")
    plt.ylim(0, 1.05)
    plt.title("Ablation: attack FAR vs enrollment size")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_features(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    melted = df.melt(
        id_vars=["feature_set", "method", "classifier_type", "subject"],
        value_vars=["far_human", "far_attack"],
        var_name="far_type",
        value_name="far",
    )
    melted["far_type"] = melted["far_type"].map(
        {"far_human": "Human impostor", "far_attack": "Synthetic attack"}
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=melted,
        x="feature_set",
        y="far",
        hue="far_type",
        errorbar="sd",
    )
    plt.ylabel("False Acceptance Rate")
    plt.xlabel("Feature set")
    plt.ylim(0, 1.05)
    plt.title("Ablation: human vs attack FAR by feature set")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run() -> None:
    args = parse_args()
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
    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    feature_sets = [f.strip() for f in args.feature_sets.split(",") if f.strip()]
    tune = bool(args.tune)
    run_id = make_run_id(f"ablation_{args.mode}", args.seed)

    print(
        f"Ablation mode={args.mode} | subjects={len(subjects)} | "
        f"classifiers={classifiers} | attacks={attacks} | run_id={run_id}"
    )

    enrollment_records: list[dict] = []
    feature_records: list[dict] = []

    if not args.skip_enrollment:
        print("\n=== Enrollment-size ablation ===")
        for enroll_sessions in ENROLLMENT_SCHEDULE:
            n_enroll = len(enroll_sessions)
            print(f"\n-- enroll sessions={enroll_sessions} --")
            for subject in subjects:
                print(f"Subject {subject}")
                result = run_single_subject(
                    df,
                    subject,
                    feature_set=DEFAULT_FEATURE_SET,
                    classifiers=classifiers,
                    attacks=attacks,
                    enroll_sessions=enroll_sessions,
                    verify_sessions=DEFAULT_VERIFY_SESSIONS,
                    impostor_samples=args.impostor_samples,
                    n_synthetic=args.n_synthetic,
                    tune_hyperparameters=tune,
                    vae_epochs=args.vae_epochs,
                    gan_epochs=args.gan_epochs,
                    device=args.device,
                    seed=args.seed,
                    save_artifacts=False,
                    mode=args.mode,
                    run_id=run_id,
                    split_mode="fixed",
                )
                for rec in result.attack_records:
                    enrollment_records.append(rec)

        enroll_df = pd.DataFrame(enrollment_records)
        enroll_path = RESULTS_DIR / "ablation_enrollment.csv"
        enroll_df.to_csv(enroll_path, index=False)
        print(f"Saved enrollment ablation → {enroll_path}")
        _plot_enrollment(enroll_df, FIGURES_DIR / "ablation_enrollment.png")
        print(f"Saved figure → {FIGURES_DIR / 'ablation_enrollment.png'}")

    if not args.skip_features:
        print("\n=== Feature-set ablation ===")
        for feature_set in feature_sets:
            print(f"\n-- feature_set={feature_set} --")
            for subject in subjects:
                print(f"Subject {subject}")
                result = run_single_subject(
                    df,
                    subject,
                    feature_set=feature_set,
                    classifiers=classifiers,
                    attacks=attacks,
                    enroll_sessions=(1, 2, 3, 4),
                    verify_sessions=DEFAULT_VERIFY_SESSIONS,
                    impostor_samples=args.impostor_samples,
                    n_synthetic=args.n_synthetic,
                    tune_hyperparameters=tune,
                    vae_epochs=args.vae_epochs,
                    gan_epochs=args.gan_epochs,
                    device=args.device,
                    seed=args.seed,
                    save_artifacts=False,
                    mode=args.mode,
                    run_id=run_id,
                    split_mode="fixed",
                )
                for rec in result.attack_records:
                    feature_records.append(rec)

        feat_df = pd.DataFrame(feature_records)
        feat_path = RESULTS_DIR / "ablation_features.csv"
        feat_df.to_csv(feat_path, index=False)
        print(f"Saved feature ablation → {feat_path}")
        _plot_features(feat_df, FIGURES_DIR / "ablation_features.png")
        print(f"Saved figure → {FIGURES_DIR / 'ablation_features.png'}")

    write_run_meta(
        RESULTS_DIR / "ablation_run_meta.json",
        mode=args.mode,
        run_id=run_id,
        seed=args.seed,
        subjects=subjects,
        feature_set=DEFAULT_FEATURE_SET,
        classifiers=classifiers,
        attacks=attacks,
        split_mode="fixed",
        enroll_sessions=(1, 2, 3, 4),
        verify_sessions=DEFAULT_VERIFY_SESSIONS,
        n_synthetic=args.n_synthetic,
        vae_epochs=args.vae_epochs,
        gan_epochs=args.gan_epochs,
        extra={
            "ablation": {
                "enrollment_schedule": [list(s) for s in ENROLLMENT_SCHEDULE],
                "feature_sets": feature_sets,
                "skip_enrollment": args.skip_enrollment,
                "skip_features": args.skip_features,
            },
            "tune_hyperparameters": tune,
        },
    )
    print(f"Saved ablation metadata → {RESULTS_DIR / 'ablation_run_meta.json'}")
    print("Ablation done.")


if __name__ == "__main__":
    run()
