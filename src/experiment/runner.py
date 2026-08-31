"""Shared experiment helpers for pilot, full, LOSO, and ablation runs."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.adversarial.synthesize import (
    generate_impersonation_samples,
    save_synthetic_samples,
)
from src.config import (
    DEFAULT_ENROLL_SESSIONS,
    DEFAULT_FEATURE_SET,
    DEFAULT_IMPOSTOR_TRAIN_SAMPLES,
    DEFAULT_SYNTHETIC_SAMPLES,
    DEFAULT_VERIFY_SESSIONS,
    GAN_DEFAULTS,
    MODELS_DIR,
    PILOT_SUBJECT_COUNT,
    RANDOM_SEED,
    RESULTS_DIR,
    SYNTHETIC_DIR,
    VAE_DEFAULTS,
)
from src.data.loader import list_subjects
from src.data.splits import (
    SubjectSplit,
    build_subject_split,
    build_training_dataset,
    build_verification_dataset,
)
from src.evaluation.metrics import (
    AttackMetrics,
    MetricsResult,
    compute_attack_metrics,
    compute_metrics,
)
from src.models.classifier import (
    predict_proba,
    save_policy_engine,
    train_policy_engine,
)
from src.models.gan import save_gan
from src.models.vae import save_vae


@dataclass
class SubjectRunResult:
    """Outputs from evaluating one subject (optionally one LOSO fold)."""

    baseline_metrics: list[MetricsResult] = field(default_factory=list)
    baseline_records: list[dict[str, Any]] = field(default_factory=list)
    attack_rows: list[AttackMetrics] = field(default_factory=list)
    attack_records: list[dict[str, Any]] = field(default_factory=list)


def resolve_subjects(
    df: pd.DataFrame,
    *,
    mode: str = "pilot",
    subjects_arg: str | None = None,
    subjects_explicit: bool = False,
) -> list[str]:
    """
    Resolve the subject cohort.

    If subjects_arg was set explicitly on the CLI, it wins over --mode.
    Otherwise pilot → first PILOT_SUBJECT_COUNT subjects; full → all.
    """
    all_subjects = list_subjects(df)
    if subjects_explicit and subjects_arg is not None:
        return _parse_subjects_arg(all_subjects, subjects_arg)
    if mode == "full":
        return all_subjects
    if mode == "pilot":
        return all_subjects[:PILOT_SUBJECT_COUNT]
    raise ValueError(f"Unknown mode: {mode!r} (expected 'pilot' or 'full')")


def _parse_subjects_arg(all_subjects: list[str], subjects_arg: str) -> list[str]:
    if subjects_arg.strip().lower() == "all":
        return all_subjects
    if "," in subjects_arg:
        requested = [s.strip() for s in subjects_arg.split(",") if s.strip()]
        missing = [s for s in requested if s not in all_subjects]
        if missing:
            raise ValueError(f"Unknown subjects: {missing}")
        return requested
    n = int(subjects_arg)
    if n < 1:
        raise ValueError("--subjects must be >= 1")
    return all_subjects[:n]


def run_single_subject(
    df: pd.DataFrame,
    subject: str,
    *,
    feature_set: str = DEFAULT_FEATURE_SET,
    classifiers: list[str],
    attacks: list[str],
    enroll_sessions: tuple[int, ...] = DEFAULT_ENROLL_SESSIONS,
    verify_sessions: tuple[int, ...] = DEFAULT_VERIFY_SESSIONS,
    split: SubjectSplit | None = None,
    impostor_samples: int = DEFAULT_IMPOSTOR_TRAIN_SAMPLES,
    n_synthetic: int = DEFAULT_SYNTHETIC_SAMPLES,
    tune_hyperparameters: bool = True,
    vae_epochs: int | None = None,
    gan_epochs: int | None = None,
    device: str = "cpu",
    seed: int = RANDOM_SEED,
    save_artifacts: bool = True,
    mode: str = "pilot",
    run_id: str = "",
    split_mode: str = "fixed",
    verbose: bool = True,
) -> SubjectRunResult:
    """
    Train Policy Engines for one subject, score verification, optionally attack.

    DOES: enroll → train PE → human metrics → (optional) VAE/GAN synthesis → attack FAR.
    WHY: Shared path for main experiment, LOSO folds, and ablation scripts.
    """
    if split is None:
        split = build_subject_split(
            df,
            subject,
            feature_set=feature_set,
            enroll_sessions=enroll_sessions,
            verify_sessions=verify_sessions,
        )

    X_train, y_train = build_training_dataset(
        split,
        impostor_samples_per_user=impostor_samples,
        random_state=seed,
    )
    X_verify, y_verify = build_verification_dataset(split)
    n_genuine = len(split.X_verify)

    result = SubjectRunResult()
    meta_tags = {
        "mode": mode,
        "run_id": run_id,
        "feature_set": feature_set,
        "split_mode": split_mode,
        "enroll_sessions": ",".join(str(s) for s in split.enroll_sessions),
        "verify_sessions": ",".join(str(s) for s in split.verify_sessions),
        "verify_session": split.verify_session,
        "n_enroll_sessions": len(split.enroll_sessions),
    }

    for clf in classifiers:
        if verbose:
            print(f"  Training Policy Engine ({clf})...")
        engine = train_policy_engine(
            X_train,
            y_train,
            subject=subject,
            feature_set=feature_set,
            classifier_type=clf,
            tune_hyperparameters=tune_hyperparameters,
            random_state=seed,
        )
        if save_artifacts:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            save_policy_engine(engine)

        scores = predict_proba(engine, X_verify)
        metrics = compute_metrics(y_verify, scores, subject=subject)
        result.baseline_metrics.append(metrics)
        result.baseline_records.append(
            {
                **asdict(metrics),
                "classifier_type": clf,
                **meta_tags,
            }
        )
        if verbose:
            print(
                f"  Human baseline [{clf}]: TAR={metrics.tar:.3f} "
                f"FAR={metrics.far:.3f} EER={metrics.eer:.3f}"
            )

        genuine_scores = scores[:n_genuine]
        impostor_scores = scores[n_genuine:]

        for method in attacks:
            if verbose:
                print(f"  Synthesizing {method.upper()} impersonation samples...")
            train_kwargs: dict[str, Any] = {}
            if method == "vae":
                train_kwargs["epochs"] = (
                    vae_epochs if vae_epochs is not None else VAE_DEFAULTS["epochs"]
                )
            elif method == "gan":
                train_kwargs["epochs"] = (
                    gan_epochs if gan_epochs is not None else GAN_DEFAULTS["epochs"]
                )

            samples, trained = generate_impersonation_samples(
                split.X_enroll,
                subject=subject,
                feature_set=feature_set,
                method=method,
                n_samples=n_synthetic,
                device=device,
                train_kwargs=train_kwargs,
            )
            if save_artifacts:
                SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
                save_synthetic_samples(
                    samples,
                    subject=subject,
                    feature_set=feature_set,
                    method=method,
                )
                if method == "vae":
                    save_vae(trained)
                else:
                    save_gan(trained)

            synth_scores = predict_proba(engine, samples)
            attack = compute_attack_metrics(
                subject=subject,
                method=method,
                classifier_type=clf,
                genuine_scores=genuine_scores,
                human_impostor_scores=impostor_scores,
                synthetic_scores=synth_scores,
                threshold=metrics.threshold,
            )
            result.attack_rows.append(attack)
            result.attack_records.append({**asdict(attack), **meta_tags})
            if verbose:
                print(
                    f"  Attack [{method}/{clf}]: TAR={attack.tar:.3f} "
                    f"FAR_human={attack.far_human:.3f} "
                    f"FAR_attack={attack.far_attack:.3f}"
                )

    return result


def make_run_id(mode: str, seed: int) -> str:
    """Build a short run identifier for tagging CSV rows."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{mode}_{seed}_{stamp}"


def git_commit_hash() -> str | None:
    """Return current git commit hash if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def package_versions() -> dict[str, str]:
    """Capture pinned runtime package versions for run_meta."""
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "sklearn", "torch", "scipy"):
        try:
            if name == "sklearn":
                import sklearn

                versions["sklearn"] = sklearn.__version__
            else:
                mod = __import__(name)
                versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not_installed"
    return versions


def write_run_meta(
    path: Path,
    *,
    mode: str,
    run_id: str,
    seed: int,
    subjects: list[str],
    feature_set: str,
    classifiers: list[str],
    attacks: list[str],
    split_mode: str,
    enroll_sessions: tuple[int, ...],
    verify_sessions: tuple[int, ...],
    n_synthetic: int,
    vae_epochs: int | None = None,
    gan_epochs: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write enriched reproducibility metadata for a results folder."""
    meta: dict[str, Any] = {
        "mode": mode,
        "run_id": run_id,
        "seed": seed,
        "subjects": subjects,
        "n_subjects": len(subjects),
        "feature_set": feature_set,
        "classifiers": classifiers,
        "attacks": attacks,
        "split_mode": split_mode,
        "enroll_sessions": list(enroll_sessions),
        "verify_sessions": list(verify_sessions),
        "n_synthetic": n_synthetic,
        "vae_hyperparams": {
            **VAE_DEFAULTS,
            **({"epochs": vae_epochs} if vae_epochs is not None else {}),
        },
        "gan_hyperparams": {
            **GAN_DEFAULTS,
            **({"epochs": gan_epochs} if gan_epochs is not None else {}),
        },
        "package_versions": package_versions(),
        "git_commit": git_commit_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_argv": list(sys.argv),
    }
    if extra:
        meta.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def ensure_results_dirs() -> None:
    """Create standard results/models/synthetic/figures directories."""
    for d in (RESULTS_DIR, MODELS_DIR, SYNTHETIC_DIR):
        d.mkdir(parents=True, exist_ok=True)
