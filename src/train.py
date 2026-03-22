"""Train TF-IDF + multiple classifiers, compare accuracy, save best model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.data_loader import get_X_y, load_news_dataframe
from src.preprocess import preprocess_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
BEST_MODEL_PATH = MODEL_DIR / "best_classifier.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

RANDOM_STATE = 42


def main() -> None:
    df = load_news_dataframe()
    X_raw, y = get_X_y(df)
    X_clean = preprocess_series(X_raw)
    # Drop empty after preprocessing (edge case)
    pairs = [(a, b) for a, b in zip(X_clean, y) if a.strip()]
    X_clean = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=25_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="saga",
        ),
        "Passive Aggressive": PassiveAggressiveClassifier(
            max_iter=1000,
            random_state=RANDOM_STATE,
            tol=1e-4,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    accuracies: dict[str, float] = {}
    reports: dict[str, str] = {}

    for name, clf in models.items():
        clf.fit(X_tr, y_train)
        pred = clf.predict(X_te)
        acc = float(accuracy_score(y_test, pred))
        accuracies[name] = acc
        reports[name] = classification_report(
            y_test,
            pred,
            target_names=["Fake", "Real"],
            digits=4,
        )
        print(f"\n{name} — accuracy: {acc:.4f}")
        print(reports[name])

    best_name = max(accuracies, key=accuracies.get)  # type: ignore[arg-type]
    best_clf = models[best_name]

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(best_clf, BEST_MODEL_PATH)

    payload = {
        "accuracies": accuracies,
        "best_model": best_name,
        "best_accuracy": accuracies[best_name],
        "n_train": len(y_train),
        "n_test": len(y_test),
        "reports": reports,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSaved vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved best model ({best_name}) -> {BEST_MODEL_PATH}")
    print(f"Metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
