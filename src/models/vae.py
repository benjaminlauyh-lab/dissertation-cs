"""Variational Autoencoder for per-user keystroke impersonation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODELS_DIR, RANDOM_SEED, VAE_DEFAULTS


class KeystrokeVAE(nn.Module):
    """MLP VAE over fixed-length keystroke timing vectors."""

    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


@dataclass
class TrainedVAE:
    """Trained per-user VAE plus feature scaling statistics."""

    subject: str
    feature_set: str
    model: KeystrokeVAE
    feature_mean: np.ndarray
    feature_std: np.ndarray
    loss_history: list[float]


def train_vae(
    X_enroll: np.ndarray,
    *,
    subject: str,
    feature_set: str,
    latent_dim: int = VAE_DEFAULTS["latent_dim"],
    hidden_dim: int = VAE_DEFAULTS["hidden_dim"],
    epochs: int = VAE_DEFAULTS["epochs"],
    batch_size: int = VAE_DEFAULTS["batch_size"],
    learning_rate: float = VAE_DEFAULTS["learning_rate"],
    beta: float = VAE_DEFAULTS["beta"],
    device: str = "cpu",
    random_state: int = RANDOM_SEED,
) -> TrainedVAE:
    """Train a VAE on one user's enrollment keystroke vectors."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    feature_mean = X_enroll.mean(axis=0)
    feature_std = X_enroll.std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    X_scaled = (X_enroll - feature_mean) / feature_std

    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=min(batch_size, len(tensor)),
        shuffle=True,
    )

    model = KeystrokeVAE(
        input_dim=X_enroll.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            recon_loss = nn.functional.mse_loss(recon, batch, reduction="sum") / batch.size(0)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        loss_history.append(epoch_loss / len(tensor))

    return TrainedVAE(
        subject=subject,
        feature_set=feature_set,
        model=model.cpu(),
        feature_mean=feature_mean,
        feature_std=feature_std,
        loss_history=loss_history,
    )


def generate_vae_samples(
    trained: TrainedVAE,
    n_samples: int,
    *,
    device: str = "cpu",
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Sample synthetic keystroke vectors from a trained VAE."""
    torch.manual_seed(random_state)
    model = trained.model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim, device=device)
        # Decode in scaled space, then invert scaling and clip to non-negative timings.
        scaled = model.decode(z).cpu().numpy()
    samples = scaled * trained.feature_std + trained.feature_mean
    return np.clip(samples, a_min=0.0, a_max=None)


def save_vae(trained: TrainedVAE, path: Path | str | None = None) -> Path:
    """Save VAE weights and scaling stats."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = MODELS_DIR / f"vae_{trained.subject}_{trained.feature_set}.pt"
    path = Path(path)
    torch.save(
        {
            "subject": trained.subject,
            "feature_set": trained.feature_set,
            "state_dict": trained.model.state_dict(),
            "input_dim": trained.model.input_dim,
            "latent_dim": trained.model.latent_dim,
            "feature_mean": trained.feature_mean,
            "feature_std": trained.feature_std,
            "loss_history": trained.loss_history,
        },
        path,
    )
    return path


def load_vae(path: Path | str, hidden_dim: int = VAE_DEFAULTS["hidden_dim"]) -> TrainedVAE:
    """Load a previously saved VAE."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = KeystrokeVAE(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"],
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return TrainedVAE(
        subject=checkpoint["subject"],
        feature_set=checkpoint["feature_set"],
        model=model,
        feature_mean=checkpoint["feature_mean"],
        feature_std=checkpoint["feature_std"],
        loss_history=checkpoint.get("loss_history", []),
    )
