"""Adversarial keystroke synthesis utilities for UEBA stress-testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_SYNTHETIC_SAMPLES, SYNTHETIC_DIR
from src.models.gan import TrainedGAN, generate_gan_samples, train_gan
from src.models.vae import TrainedVAE, generate_vae_samples, train_vae


def generate_impersonation_samples(
    X_enroll: np.ndarray,
    *,
    subject: str,
    feature_set: str,
    method: str,
    n_samples: int = DEFAULT_SYNTHETIC_SAMPLES,
    device: str = "cpu",
    train_kwargs: dict | None = None,
) -> tuple[np.ndarray, TrainedVAE | TrainedGAN]:
    """
    Train a generative model on enrollment data and produce synthetic samples.

    method: 'vae' or 'gan'
    """
    train_kwargs = train_kwargs or {}
    method = method.lower()

    if method == "vae":
        trained = train_vae(
            X_enroll,
            subject=subject,
            feature_set=feature_set,
            device=device,
            **train_kwargs,
        )
        samples = generate_vae_samples(trained, n_samples, device=device)
        return samples, trained

    if method == "gan":
        trained = train_gan(
            X_enroll,
            subject=subject,
            feature_set=feature_set,
            device=device,
            **train_kwargs,
        )
        samples = generate_gan_samples(trained, n_samples, device=device)
        return samples, trained

    raise ValueError(f"Unknown synthesis method '{method}'. Use 'vae' or 'gan'.")


def save_synthetic_samples(
    samples: np.ndarray,
    *,
    subject: str,
    feature_set: str,
    method: str,
    path: Path | str | None = None,
) -> Path:
    """Persist synthetic feature matrix as CSV for auditing / dissertation figures."""
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = SYNTHETIC_DIR / f"{method}_{subject}_{feature_set}.csv"
    path = Path(path)
    columns = [f"f{i}" for i in range(samples.shape[1])]
    pd.DataFrame(samples, columns=columns).to_csv(path, index=False)
    return path
