"""Policy Engine classifiers for behavioral biometric verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import CLASSIFIER_TYPES, DEFAULT_CLASSIFIER, MODELS_DIR, RANDOM_SEED


@dataclass
class PolicyEngine:
    """Trained UEBA-style classifier for one authorized subject."""

    subject: str
    feature_set: str
    classifier_type: str
    pipeline: Pipeline


def train_policy_engine(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    subject: str,
    feature_set: str,
    classifier_type: str = DEFAULT_CLASSIFIER,
    tune_hyperparameters: bool = True,
    random_state: int = RANDOM_SEED,
) -> PolicyEngine:
    """Train and return a Policy Engine for one authorized user."""
    _validate_classifier_type(classifier_type)

    if tune_hyperparameters:
        pipeline = _tune_pipeline(X_train, y_train, classifier_type, random_state)
    else:
        pipeline = _build_pipeline(classifier_type, random_state)
        pipeline.fit(X_train, y_train)

    return PolicyEngine(
        subject=subject,
        feature_set=feature_set,
        classifier_type=classifier_type,
        pipeline=pipeline,
    )


def save_policy_engine(engine: PolicyEngine, path: Path | str | None = None) -> Path:
    """Persist a trained Policy Engine to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = MODELS_DIR / f"pe_{engine.subject}_{engine.feature_set}_{engine.classifier_type}.joblib"
    path = Path(path)
    joblib.dump(engine, path)
    return path


def load_policy_engine(path: Path | str) -> PolicyEngine:
    """Load a trained Policy Engine from disk."""
    return joblib.load(path)


def predict_proba(engine: PolicyEngine, X: np.ndarray) -> np.ndarray:
    """Return P(authorized) scores for verification samples."""
    if hasattr(engine.pipeline, "predict_proba"):
        return engine.pipeline.predict_proba(X)[:, 1]
    # decision_function fallback for older sklearn SVM configs
    scores = engine.pipeline.decision_function(X)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


def _validate_classifier_type(classifier_type: str) -> None:
    if classifier_type not in CLASSIFIER_TYPES:
        available = ", ".join(CLASSIFIER_TYPES)
        raise ValueError(
            f"Unknown classifier '{classifier_type}'. Available: {available}"
        )


def _build_pipeline(classifier_type: str, random_state: int) -> Pipeline:
    if classifier_type == "svm":
        # sklearn >= 1.9 deprecates SVC(probability=True); calibrate separately.
        estimator = CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=random_state),
            method="sigmoid",
            cv=3,
        )
    else:
        estimator = RandomForestClassifier(random_state=random_state)

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def _tune_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_type: str,
    random_state: int,
) -> Pipeline:
    if classifier_type == "svm":
        pipeline = _build_pipeline("svm", random_state)
        param_grid = {
            "classifier__estimator__C": [0.1, 1.0, 10.0],
            "classifier__estimator__gamma": ["scale", "auto", 0.01, 0.1],
        }
    else:
        pipeline = _build_pipeline("random_forest", random_state)
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
        }

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_
