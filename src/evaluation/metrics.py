"""Biometric evaluation metrics for the Policy Engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class MetricsResult:
    """Human verification metrics for one subject."""

    subject: str
    tar: float
    far: float
    eer: float
    accuracy: float
    auc: float
    threshold: float


@dataclass(frozen=True)
class AttackMetrics:
    """Impersonation attack metrics at a fixed PE threshold."""

    subject: str
    method: str
    classifier_type: str
    tar: float
    far_human: float
    far_attack: float
    threshold: float
    n_synthetic: int


@dataclass(frozen=True)
class CISummary:
    """Mean / std / 95% confidence interval for a numeric series."""

    mean: float
    std: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class PairedFARTest:
    """Paired comparison of human FAR vs attack FAR."""

    method: str
    classifier_type: str
    n_pairs: int
    mean_far_human: float
    mean_far_attack: float
    mean_delta: float
    test_name: str
    statistic: float
    p_value: float


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    subject: str,
) -> MetricsResult:
    """
    Compute TAR, FAR, EER, accuracy, and AUC.

    Positive class (1) = authorized user.
    Negative class (0) = impostor.
    """
    if len(np.unique(y_true)) < 2:
        raise ValueError("Metrics require both positive and negative verification samples.")

    threshold = _equal_error_threshold(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    impostor_total = tn + fp
    genuine_total = fn + tp

    tar = tp / genuine_total if genuine_total else 0.0
    far = fp / impostor_total if impostor_total else 0.0
    eer = _equal_error_rate(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)

    return MetricsResult(
        subject=subject,
        tar=float(tar),
        far=float(far),
        eer=float(eer),
        accuracy=float(accuracy),
        auc=float(auc),
        threshold=float(threshold),
    )


def compute_attack_metrics(
    *,
    subject: str,
    method: str,
    classifier_type: str,
    genuine_scores: np.ndarray,
    human_impostor_scores: np.ndarray,
    synthetic_scores: np.ndarray,
    threshold: float,
) -> AttackMetrics:
    """
    Evaluate FAR under synthetic impersonation at a fixed operating threshold.

    TAR and FAR_human use the same threshold so attack FAR is comparable.
    """
    tar = float(np.mean(genuine_scores >= threshold)) if len(genuine_scores) else 0.0
    far_human = (
        float(np.mean(human_impostor_scores >= threshold))
        if len(human_impostor_scores)
        else 0.0
    )
    far_attack = (
        float(np.mean(synthetic_scores >= threshold)) if len(synthetic_scores) else 0.0
    )
    return AttackMetrics(
        subject=subject,
        method=method,
        classifier_type=classifier_type,
        tar=tar,
        far_human=far_human,
        far_attack=far_attack,
        threshold=float(threshold),
        n_synthetic=int(len(synthetic_scores)),
    )


def summarize_with_ci(
    series: np.ndarray | list[float] | pd.Series,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> CISummary:
    """
    Summarize a numeric series as mean, std, and 95% CI.

    Uses a Student-t interval when n >= 10; otherwise percentile bootstrap.
    """
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    n = int(len(values))
    if n == 0:
        return CISummary(mean=0.0, std=0.0, ci_low=0.0, ci_high=0.0, n=0)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    if n == 1:
        return CISummary(mean=mean, std=0.0, ci_low=mean, ci_high=mean, n=1)

    if n < 10:
        rng = np.random.default_rng(random_state)
        boots = rng.choice(values, size=(n_bootstrap, n), replace=True).mean(axis=1)
        alpha = 1.0 - confidence
        ci_low = float(np.quantile(boots, alpha / 2))
        ci_high = float(np.quantile(boots, 1.0 - alpha / 2))
    else:
        se = std / np.sqrt(n)
        ci_low, ci_high = stats.t.interval(confidence, df=n - 1, loc=mean, scale=se)
        ci_low, ci_high = float(ci_low), float(ci_high)

    return CISummary(mean=mean, std=std, ci_low=ci_low, ci_high=ci_high, n=n)


def paired_far_test(
    far_human: np.ndarray | list[float],
    far_attack: np.ndarray | list[float],
    *,
    method: str = "",
    classifier_type: str = "",
) -> PairedFARTest:
    """
    Test whether attack FAR exceeds human FAR on matched per-subject pairs.

    Prefers Wilcoxon signed-rank; falls back to paired t-test when Wilcoxon
    cannot be computed (e.g. all differences zero or n too small).
    """
    h = np.asarray(far_human, dtype=float)
    a = np.asarray(far_attack, dtype=float)
    if len(h) != len(a):
        raise ValueError("far_human and far_attack must have the same length")
    n = len(h)
    mean_h = float(np.mean(h)) if n else 0.0
    mean_a = float(np.mean(a)) if n else 0.0
    mean_delta = mean_a - mean_h

    if n < 2:
        return PairedFARTest(
            method=method,
            classifier_type=classifier_type,
            n_pairs=n,
            mean_far_human=mean_h,
            mean_far_attack=mean_a,
            mean_delta=mean_delta,
            test_name="none",
            statistic=float("nan"),
            p_value=float("nan"),
        )

    diffs = a - h
    test_name = "wilcoxon"
    try:
        if np.allclose(diffs, 0.0):
            raise ValueError("all differences zero")
        statistic, p_value = stats.wilcoxon(a, h, alternative="greater", zero_method="wilcox")
        statistic = float(statistic)
        p_value = float(p_value)
    except ValueError:
        test_name = "paired_t"
        statistic, p_value = stats.ttest_rel(a, h, alternative="greater")
        statistic = float(statistic)
        p_value = float(p_value)

    return PairedFARTest(
        method=method,
        classifier_type=classifier_type,
        n_pairs=n,
        mean_far_human=mean_h,
        mean_far_attack=mean_a,
        mean_delta=mean_delta,
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
    )


def summarize_attack_results(results: list[AttackMetrics]) -> pd.DataFrame:
    """Aggregate per-subject attack metrics into a summary table with CIs."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in results])
    rows = []
    for (method, clf), group in df.groupby(["method", "classifier_type"]):
        tar_ci = summarize_with_ci(group["tar"])
        far_h_ci = summarize_with_ci(group["far_human"])
        far_a_ci = summarize_with_ci(group["far_attack"])
        rows.append(
            {
                "method": method,
                "classifier_type": clf,
                "n_subjects": int(group["subject"].nunique()),
                "tar_mean": tar_ci.mean,
                "tar_std": tar_ci.std,
                "tar_ci_low": tar_ci.ci_low,
                "tar_ci_high": tar_ci.ci_high,
                "far_human_mean": far_h_ci.mean,
                "far_human_std": far_h_ci.std,
                "far_human_ci_low": far_h_ci.ci_low,
                "far_human_ci_high": far_h_ci.ci_high,
                "far_attack_mean": far_a_ci.mean,
                "far_attack_std": far_a_ci.std,
                "far_attack_ci_low": far_a_ci.ci_low,
                "far_attack_ci_high": far_a_ci.ci_high,
            }
        )
    return pd.DataFrame(rows)


def build_statistical_summary(results: list[AttackMetrics]) -> pd.DataFrame:
    """Flat statistical summary suitable for results/statistical_summary.csv."""
    return summarize_attack_results(results)


def build_hypothesis_tests(results: list[AttackMetrics]) -> pd.DataFrame:
    """Paired FAR tests per (method, classifier) for results/hypothesis_tests.csv."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in results])
    rows = []
    for (method, clf), group in df.groupby(["method", "classifier_type"]):
        # One pair per subject (average if multiple folds present)
        per_subject = group.groupby("subject", as_index=False).agg(
            far_human=("far_human", "mean"),
            far_attack=("far_attack", "mean"),
        )
        test = paired_far_test(
            per_subject["far_human"].to_numpy(),
            per_subject["far_attack"].to_numpy(),
            method=str(method),
            classifier_type=str(clf),
        )
        rows.append(test.__dict__)
    return pd.DataFrame(rows)


def _equal_error_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    return float(thresholds[eer_index])


def _equal_error_rate(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    return float(np.min(np.abs(fpr - fnr)))
