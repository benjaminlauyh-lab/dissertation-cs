#!/usr/bin/env python3
"""Generate dissertation figures from saved experiment CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402


def _plot_far_with_ci(detail: pd.DataFrame, summary_path: Path, out: Path) -> None:
    """Barplot of mean FAR with 95% CI whiskers when statistical_summary exists."""
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        rows = []
        for _, row in summary.iterrows():
            label = f"{str(row['method']).upper()}\n{row['classifier_type']}"
            for far_type, mean_col, lo_col, hi_col in (
                ("Human impostor", "far_human_mean", "far_human_ci_low", "far_human_ci_high"),
                ("Synthetic attack", "far_attack_mean", "far_attack_ci_low", "far_attack_ci_high"),
            ):
                if mean_col not in summary.columns:
                    continue
                mean = float(row[mean_col])
                if lo_col in summary.columns:
                    err_lo = max(0.0, mean - float(row[lo_col]))
                    err_hi = max(0.0, float(row[hi_col]) - mean)
                else:
                    err_lo = err_hi = 0.0
                rows.append(
                    {
                        "group": label,
                        "far_type": far_type,
                        "far": mean,
                        "err_lo": err_lo,
                        "err_hi": err_hi,
                    }
                )
        plot_df = pd.DataFrame(rows)
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=plot_df, x="group", y="far", hue="far_type")
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
            )
        plt.title("UEBA FAR: Human Impostors vs Generative Impersonation (mean ± 95% CI)")
    else:
        melted = detail.melt(
            id_vars=["subject", "method", "classifier_type"],
            value_vars=["far_human", "far_attack"],
            var_name="far_type",
            value_name="far",
        )
        melted["far_type"] = melted["far_type"].map(
            {
                "far_human": "Human impostor",
                "far_attack": "Synthetic attack",
            }
        )
        melted["group"] = (
            melted["method"].str.upper() + "\n" + melted["classifier_type"]
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(data=melted, x="group", y="far", hue="far_type", errorbar="sd")
        plt.title("UEBA FAR: Human Impostors vs Generative Impersonation")

    plt.ylabel("False Acceptance Rate")
    plt.xlabel("")
    plt.ylim(0, 1.05)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    baseline_path = RESULTS_DIR / "human_baseline.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, col, title in zip(
            axes,
            ["tar", "far", "eer"],
            ["True Acceptance Rate", "FAR (human impostors)", "Equal Error Rate"],
        ):
            if "classifier_type" in baseline.columns:
                sns.boxplot(data=baseline, x="classifier_type", y=col, ax=ax)
            else:
                sns.boxplot(data=baseline, y=col, ax=ax)
            ax.set_title(title)
            ax.set_ylim(0, 1)
        fig.suptitle("Human-only Policy Engine baseline")
        fig.tight_layout()
        out = FIGURES_DIR / "human_baseline.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Wrote {out}")

        if "classifier_type" in baseline.columns:
            summary = baseline.groupby("classifier_type")[["tar", "far", "eer"]].agg(
                ["mean", "std"]
            )
            summary.to_csv(RESULTS_DIR / "human_baseline_summary.csv")
            print(summary)

    attack_path = RESULTS_DIR / "attack_results.csv"
    if attack_path.exists():
        detail = pd.read_csv(attack_path)
        _plot_far_with_ci(
            detail,
            RESULTS_DIR / "statistical_summary.csv",
            FIGURES_DIR / "far_comparison.png",
        )

        detail = detail.copy()
        detail["far_delta"] = detail["far_attack"] - detail["far_human"]
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=detail, x="method", y="far_delta", hue="classifier_type")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.ylabel("Δ FAR (attack − human)")
        plt.title("Increase in FAR under generative impersonation")
        plt.tight_layout()
        out = FIGURES_DIR / "far_delta.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Wrote {out}")

        plt.figure(figsize=(8, 4))
        sns.boxplot(data=detail, x="method", y="tar", hue="classifier_type")
        plt.ylim(0, 1)
        plt.ylabel("TAR at EER threshold")
        plt.title("Genuine acceptance retained at attack operating point")
        plt.tight_layout()
        out = FIGURES_DIR / "tar_retention.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Wrote {out}")
    else:
        print("No attack_results.csv yet — run scripts/run_experiment.py first.")

    # Optional ablation figures from saved CSVs
    enroll_path = RESULTS_DIR / "ablation_enrollment.csv"
    if enroll_path.exists():
        enroll = pd.read_csv(enroll_path)
        if not enroll.empty and "n_enroll_sessions" in enroll.columns:
            agg = (
                enroll.groupby(
                    ["n_enroll_sessions", "method", "classifier_type"], as_index=False
                ).agg(far_attack=("far_attack", "mean"))
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
            out = FIGURES_DIR / "ablation_enrollment.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"Wrote {out}")

    feat_path = RESULTS_DIR / "ablation_features.csv"
    if feat_path.exists():
        feat = pd.read_csv(feat_path)
        if not feat.empty and "feature_set" in feat.columns:
            melted = feat.melt(
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
            out = FIGURES_DIR / "ablation_features.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
