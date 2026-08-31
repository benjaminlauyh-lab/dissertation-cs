"""Keystroke feature extraction with configurable feature sets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_SETS, METADATA_COLUMNS


def list_feature_sets() -> dict[str, str]:
    """Return supported feature set names and descriptions."""
    return dict(FEATURE_SETS)


def get_feature_columns(feature_set: str) -> list[str]:
    """Return raw CSV column names used by a feature set."""
    _validate_feature_set(feature_set)

    if feature_set == "hold":
        return _columns_with_prefix("H.")
    if feature_set == "flight":
        return _columns_with_prefix("DD.")
    if feature_set == "hold_flight":
        return _columns_with_prefix("H.") + _columns_with_prefix("DD.")
    if feature_set == "full":
        return (
            _columns_with_prefix("H.")
            + _columns_with_prefix("DD.")
            + _columns_with_prefix("UD.")
        )
    if feature_set == "aggregated":
        return []  # computed dynamically
    raise ValueError(f"Unsupported feature set: {feature_set}")


def extract_features(df: pd.DataFrame, feature_set: str = "hold_flight") -> np.ndarray:
    """
    Convert keystroke repetitions into a numeric feature matrix.

    Each row is one typing repetition. Feature sets are configurable so the
    Policy Engine can be trained on different telemetry views.
    """
    _validate_feature_set(feature_set)

    if feature_set == "aggregated":
        return _extract_aggregated_features(df)

    columns = get_feature_columns(feature_set)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing feature columns: {missing}")

    features = df[columns].to_numpy(dtype=float)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_aggregated_features(df: pd.DataFrame) -> np.ndarray:
    hold_cols = _columns_with_prefix("H.")
    flight_cols = _columns_with_prefix("DD.")

    hold = df[hold_cols].to_numpy(dtype=float)
    flight = df[flight_cols].to_numpy(dtype=float)

    aggregated = np.column_stack(
        [
            hold.mean(axis=1),
            hold.std(axis=1),
            flight.mean(axis=1),
            flight.std(axis=1),
        ]
    )
    return np.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)


def _columns_with_prefix(prefix: str) -> list[str]:
    # Canonical CMU column ordering for reproducibility across runs.
    canonical = [
        "period",
        "t",
        "i",
        "e",
        "five",
        "Shift.r",
        "o",
        "a",
        "n",
        "l",
        "Return",
    ]
    if prefix == "H.":
        return [f"H.{name}" for name in canonical]

    dd_map = {
        "period": "period.t",
        "t": "t.i",
        "i": "i.e",
        "e": "e.five",
        "five": "five.Shift.r",
        "Shift.r": "Shift.r.o",
        "o": "o.a",
        "a": "a.n",
        "n": "n.l",
        "l": "l.Return",
    }
    ud_map = dict(dd_map)

    # 10 digraph columns (period.t … l.Return); Hold has 11 including H.Return.
    digraph_keys = list(dd_map.keys())
    if prefix == "DD.":
        return [f"DD.{dd_map[name]}" for name in digraph_keys]
    if prefix == "UD.":
        return [f"UD.{ud_map[name]}" for name in digraph_keys]

    raise ValueError(f"Unsupported prefix: {prefix}")


def _validate_feature_set(feature_set: str) -> None:
    if feature_set not in FEATURE_SETS:
        available = ", ".join(FEATURE_SETS)
        raise ValueError(f"Unknown feature set '{feature_set}'. Available: {available}")
