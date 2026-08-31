"""WGAN-GP synthesizer for per-user keystroke impersonation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset

from src.config import GAN_DEFAULTS, MODELS_DIR, RANDOM_SEED


class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainedGAN:
    """Trained per-user WGAN-GP plus feature scaling statistics."""

    subject: str
    feature_set: str
    generator: Generator
    feature_mean: np.ndarray
    feature_std: np.ndarray
    loss_history: list[dict[str, float]]


def _gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: str,
) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolates)
    gradients = grad(
        outputs=scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_gan(
    X_enroll: np.ndarray,
    *,
    subject: str,
    feature_set: str,
    latent_dim: int = GAN_DEFAULTS["latent_dim"],
    hidden_dim: int = GAN_DEFAULTS["hidden_dim"],
    epochs: int = GAN_DEFAULTS["epochs"],
    batch_size: int = GAN_DEFAULTS["batch_size"],
    learning_rate: float = GAN_DEFAULTS["learning_rate"],
    n_critic: int = GAN_DEFAULTS["n_critic"],
    lambda_gp: float = GAN_DEFAULTS["lambda_gp"],
    device: str = "cpu",
    random_state: int = RANDOM_SEED,
) -> TrainedGAN:
    """Train a WGAN-GP on one user's enrollment keystroke vectors."""
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
        drop_last=True if len(tensor) >= batch_size else False,
    )

    output_dim = X_enroll.shape[1]
    generator = Generator(latent_dim, output_dim, hidden_dim).to(device)
    critic = Critic(output_dim, hidden_dim).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))

    loss_history: list[dict[str, float]] = []
    for _ in range(epochs):
        critic_losses = []
        gen_losses = []
        for (real,) in loader:
            real = real.to(device)
            current_bs = real.size(0)

            for _critic_step in range(n_critic):
                z = torch.randn(current_bs, latent_dim, device=device)
                fake = generator(z).detach()
                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()
                gp = _gradient_penalty(critic, real, fake, device)
                loss_c = critic_fake - critic_real + lambda_gp * gp

                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()
                critic_losses.append(loss_c.item())

            z = torch.randn(current_bs, latent_dim, device=device)
            fake = generator(z)
            loss_g = -critic(fake).mean()
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            gen_losses.append(loss_g.item())

        loss_history.append(
            {
                "critic": float(np.mean(critic_losses)) if critic_losses else 0.0,
                "generator": float(np.mean(gen_losses)) if gen_losses else 0.0,
            }
        )

    return TrainedGAN(
        subject=subject,
        feature_set=feature_set,
        generator=generator.cpu(),
        feature_mean=feature_mean,
        feature_std=feature_std,
        loss_history=loss_history,
    )


def generate_gan_samples(
    trained: TrainedGAN,
    n_samples: int,
    *,
    device: str = "cpu",
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Sample synthetic keystroke vectors from a trained GAN generator."""
    torch.manual_seed(random_state)
    generator = trained.generator.to(device)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, generator.latent_dim, device=device)
        scaled = generator(z).cpu().numpy()
    samples = scaled * trained.feature_std + trained.feature_mean
    return np.clip(samples, a_min=0.0, a_max=None)


def save_gan(trained: TrainedGAN, path: Path | str | None = None) -> Path:
    """Save GAN generator weights and scaling stats."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = MODELS_DIR / f"gan_{trained.subject}_{trained.feature_set}.pt"
    path = Path(path)
    torch.save(
        {
            "subject": trained.subject,
            "feature_set": trained.feature_set,
            "state_dict": trained.generator.state_dict(),
            "latent_dim": trained.generator.latent_dim,
            "output_dim": int(trained.feature_mean.shape[0]),
            "feature_mean": trained.feature_mean,
            "feature_std": trained.feature_std,
            "loss_history": trained.loss_history,
        },
        path,
    )
    return path


def load_gan(path: Path | str, hidden_dim: int = GAN_DEFAULTS["hidden_dim"]) -> TrainedGAN:
    """Load a previously saved GAN generator."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    output_dim = int(checkpoint["feature_mean"].shape[0])
    generator = Generator(
        latent_dim=checkpoint["latent_dim"],
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    )
    generator.load_state_dict(checkpoint["state_dict"])
    return TrainedGAN(
        subject=checkpoint["subject"],
        feature_set=checkpoint["feature_set"],
        generator=generator,
        feature_mean=checkpoint["feature_mean"],
        feature_std=checkpoint["feature_std"],
        loss_history=checkpoint.get("loss_history", []),
    )
