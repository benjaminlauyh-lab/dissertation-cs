"""Subject-wise enrollment/verification splits for UEBA training."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    ALL_SESSIONS,
    DEFAULT_ENROLL_SESSIONS,
    DEFAULT_VERIFY_SESSIONS,
    RANDOM_SEED,
)
from src.data.loader import get_subject_data, list_subjects
from src.features.keystroke import extract_features


@dataclass(frozen=True)
class SubjectSplit:
    """Feature matrices for one authorized subject."""

    subject: str
    feature_set: str
    X_enroll: np.ndarray
    X_verify: np.ndarray
    X_impostor_enroll: np.ndarray
    X_impostor_verify: np.ndarray
    enroll_sessions: tuple[int, ...] = DEFAULT_ENROLL_SESSIONS
    verify_sessions: tuple[int, ...] = DEFAULT_VERIFY_SESSIONS

    @property
    def verify_session(self) -> int | None:
        """Primary held-out verify session when a single session is used."""
        if len(self.verify_sessions) == 1:
            return int(self.verify_sessions[0])
        return None


def build_subject_split(
    df: pd.DataFrame,
    subject: str,
    feature_set: str = "hold_flight",
    enroll_sessions: tuple[int, ...] = DEFAULT_ENROLL_SESSIONS,
    verify_sessions: tuple[int, ...] = DEFAULT_VERIFY_SESSIONS,
) -> SubjectSplit:
    """Build enrollment and verification matrices for one authorized user."""
    subject_df = get_subject_data(df, subject)
    impostor_subjects = [s for s in list_subjects(df) if s != subject]

    enroll_mask = subject_df["sessionIndex"].isin(enroll_sessions)
    verify_mask = subject_df["sessionIndex"].isin(verify_sessions)

    X_enroll = extract_features(subject_df.loc[enroll_mask], feature_set)
    X_verify = extract_features(subject_df.loc[verify_mask], feature_set)

    impostor_enroll_parts = []
    impostor_verify_parts = []
    for impostor in impostor_subjects:
        impostor_df = get_subject_data(df, impostor)
        impostor_enroll_parts.append(
            extract_features(
                impostor_df.loc[impostor_df["sessionIndex"].isin(enroll_sessions)],
                feature_set,
            )
        )
        impostor_verify_parts.append(
            extract_features(
                impostor_df.loc[impostor_df["sessionIndex"].isin(verify_sessions)],
                feature_set,
            )
        )

    return SubjectSplit(
        subject=subject,
        feature_set=feature_set,
        X_enroll=X_enroll,
        X_verify=X_verify,
        X_impostor_enroll=np.vstack(impostor_enroll_parts),
        X_impostor_verify=np.vstack(impostor_verify_parts),
        enroll_sessions=tuple(enroll_sessions),
        verify_sessions=tuple(verify_sessions),
    )


def build_leave_one_session_out_splits(
    df: pd.DataFrame,
    subject: str,
    feature_set: str = "hold_flight",
    sessions: tuple[int, ...] = ALL_SESSIONS,
) -> Iterator[SubjectSplit]:
    """
    Yield LOSO folds: for each held-out verify session v, enroll = all others.

    Dataset sessions are typically {1, 2, 3, 4, 5}.
    """
    for verify_session in sessions:
        enroll_sessions = tuple(s for s in sessions if s != verify_session)
        yield build_subject_split(
            df,
            subject,
            feature_set=feature_set,
            enroll_sessions=enroll_sessions,
            verify_sessions=(verify_session,),
        )


def build_training_dataset(
    split: SubjectSplit,
    impostor_samples_per_user: int | None = None,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a binary training set for the Policy Engine.

    Positive (1): authorized user's enrollment keystrokes.
    Negative (0): human impostor keystrokes from other subjects.
    """
    rng = np.random.default_rng(random_state)

    X_positive = split.X_enroll
    y_positive = np.ones(len(X_positive), dtype=int)

    X_negative = split.X_impostor_enroll
    if impostor_samples_per_user is not None:
        # Subsample impostors to balance classes if requested
        max_negative = min(len(X_negative), impostor_samples_per_user)
        indices = rng.choice(len(X_negative), size=max_negative, replace=False)
        X_negative = X_negative[indices]

    y_negative = np.zeros(len(X_negative), dtype=int)

    X_train = np.vstack([X_positive, X_negative])
    y_train = np.concatenate([y_positive, y_negative])

    shuffle_idx = rng.permutation(len(y_train))
    return X_train[shuffle_idx], y_train[shuffle_idx]


def build_verification_dataset(split: SubjectSplit) -> tuple[np.ndarray, np.ndarray]:
    """Build held-out verification data: genuine (1) vs impostor (0) human samples."""
    X_genuine = split.X_verify
    y_genuine = np.ones(len(X_genuine), dtype=int)

    X_impostor = split.X_impostor_verify
    y_impostor = np.zeros(len(X_impostor), dtype=int)

    X_verify = np.vstack([X_genuine, X_impostor])
    y_verify = np.concatenate([y_genuine, y_impostor])
    return X_verify, y_verify
