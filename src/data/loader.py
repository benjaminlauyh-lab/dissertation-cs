"""Load and validate the CMU keystroke dynamics dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    DATASET_PATH,
    EXPECTED_REPS_PER_SUBJECT,
    EXPECTED_SUBJECT_COUNT,
    METADATA_COLUMNS,
)


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the CMU DSL-StrongPasswordData CSV and run basic validation."""
    dataset_path = Path(path) if path is not None else DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Download from https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv "
            f"and place it in {dataset_path.parent}."
        )

    df = pd.read_csv(dataset_path)
    _validate_dataset(df)
    return df


def list_subjects(df: pd.DataFrame) -> list[str]:
    """Return sorted subject identifiers (e.g. s002, s003, ...)."""
    return sorted(df["subject"].unique().tolist())


def get_subject_data(df: pd.DataFrame, subject: str) -> pd.DataFrame:
    """Return all repetitions for a single subject."""
    subject_df = df[df["subject"] == subject].copy()
    if subject_df.empty:
        raise ValueError(f"Subject '{subject}' not found in dataset.")
    return subject_df


def _validate_dataset(df: pd.DataFrame) -> None:
    required_columns = set(METADATA_COLUMNS)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    subject_count = df["subject"].nunique()
    if subject_count != EXPECTED_SUBJECT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECT_COUNT} subjects, found {subject_count}."
        )

    reps_per_subject = df.groupby("subject").size()
    if not (reps_per_subject == EXPECTED_REPS_PER_SUBJECT).all():
        bad = reps_per_subject[reps_per_subject != EXPECTED_REPS_PER_SUBJECT]
        raise ValueError(
            "Unexpected repetition counts per subject: "
            f"{bad.to_dict()}"
        )
