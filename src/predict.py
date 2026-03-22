"""Load saved vectorizer + model and score new text."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from src.preprocess import preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "best_classifier.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"


def _proba_from_model(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    scores = model.decision_function(X)
    if scores.ndim == 1:
        p1 = expit(scores)
        return np.column_stack([1.0 - p1, p1])
    # Multiclass fallback
    raise ValueError("Cannot derive probabilities for this model.")


def load_artifacts(
    vectorizer_path: Path | None = None,
    model_path: Path | None = None,
):
    v_path = vectorizer_path or VECTORIZER_PATH
    m_path = model_path or MODEL_PATH
    if not v_path.exists() or not m_path.exists():
        return None, None
    vectorizer = joblib.load(v_path)
    model = joblib.load(m_path)
    return vectorizer, model


def predict_article(
    raw_text: str,
    vectorizer=None,
    model=None,
) -> tuple[str, float, np.ndarray]:
    """
    Returns (predicted_label, confidence_for_predicted_class, proba_vector).
    Labels: FAKE=0, REAL=1; displayed as 'Fake' / 'Real'.
    """
    if vectorizer is None or model is None:
        vectorizer, model = load_artifacts()
    if vectorizer is None or model is None:
        raise FileNotFoundError(
            "Trained model not found. Run: python -m src.train"
        )

    cleaned = preprocess_text(raw_text)
    X = vectorizer.transform([cleaned])
    proba = _proba_from_model(model, X)[0]
    pred = int(np.argmax(proba))
    confidence = float(proba[pred])
    name = "Real" if pred == 1 else "Fake"
    return name, confidence, proba
