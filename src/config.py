"""Central configuration for the dissertation lab pipeline."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# Reproducibility
RANDOM_SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_FILENAME = "DSL-StrongPasswordData.csv"
DATASET_PATH = DATA_RAW_DIR / DATASET_FILENAME
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
SYNTHETIC_DIR = RESULTS_DIR / "synthetic"
FIGURES_DIR = RESULTS_DIR / "figures"

# Dataset metadata
METADATA_COLUMNS = ("subject", "sessionIndex", "rep")
EXPECTED_SUBJECT_COUNT = 51
EXPECTED_REPS_PER_SUBJECT = 400
ALL_SESSIONS = (1, 2, 3, 4, 5)

# Cohort modes: pilot = first N subjects; full = all subjects
PILOT_SUBJECT_COUNT = 5
DEFAULT_MODE = "pilot"

# Supported feature sets for keystroke vectors
FEATURE_SETS = {
    "hold": "Hold times only (H.*)",
    "flight": "Flight times only (DD.*)",
    "hold_flight": "Hold + flight times (H.* + DD.*) — proposal default",
    "full": "Hold + flight + up-down times (H.* + DD.* + UD.*)",
    "aggregated": "Per-rep mean/std summary of hold and flight timings",
}
DEFAULT_FEATURE_SET = "hold_flight"

# Default session splits (5 sessions × 80 reps per subject)
DEFAULT_ENROLL_SESSIONS = (1, 2, 3, 4)
DEFAULT_VERIFY_SESSIONS = (5,)
DEFAULT_SPLIT_MODE = "fixed"  # fixed | losso

# Policy Engine classifiers
CLASSIFIER_TYPES = ("svm", "random_forest")
DEFAULT_CLASSIFIER = "svm"

# Impostor sampling for balanced PE training (none = use all impostors)
DEFAULT_IMPOSTOR_TRAIN_SAMPLES = 640

# Generative synthesizers
DEFAULT_SYNTHETIC_SAMPLES = 80
DEFAULT_DEVICE = "cpu"

VAE_DEFAULTS = {
    "latent_dim": 8,
    "hidden_dim": 64,
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "beta": 1.0,
}

GAN_DEFAULTS = {
    "latent_dim": 16,
    "hidden_dim": 64,
    "epochs": 300,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "n_critic": 5,
    "lambda_gp": 10.0,
}


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs in one place for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
