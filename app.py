"""Streamlit dashboard: classify news, show probabilities and model accuracies."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.predict import METRICS_PATH, load_artifacts, predict_article


@st.cache_resource
def _artifacts():
    return load_artifacts()


@st.cache_data
def _metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(
        page_title="Fake News Detection",
        page_icon="📰",
        layout="centered",
    )
    st.title("Fake News Detection System")
    st.caption("TF-IDF + ML — Logistic Regression, Passive Aggressive, Random Forest")

    vectorizer, model = _artifacts()
    metrics = _metrics()

    if vectorizer is None or model is None:
        st.warning(
            "No trained model found. From the project folder run:\n\n"
            "`pip install -r requirements.txt`\n\n"
            "`python -m src.train`\n\n"
            "Training downloads the public fake/real news CSV into `data/` on first run."
        )
        return

    text = st.text_area(
        "Paste a news headline or article",
        height=200,
        placeholder="Enter text to classify as Fake or Real…",
    )

    if st.button("Predict", type="primary"):
        if not text or not text.strip():
            st.error("Please enter some text.")
        else:
            try:
                label, confidence, proba = predict_article(
                    text, vectorizer=vectorizer, model=model
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")
                return

            p_fake = float(proba[0])
            p_real = float(proba[1])

            st.subheader("Prediction")
            st.metric("Label", label, help="Model's predicted class")
            st.metric(
                "Confidence (for predicted class)",
                f"{confidence * 100:.1f}%",
            )

            st.subheader("Class probabilities")
            st.write("Fake")
            st.progress(min(max(p_fake, 0.0), 1.0))
            st.write("Real")
            st.progress(min(max(p_real, 0.0), 1.0))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("P(Fake)", f"{p_fake * 100:.1f}%")
            with col2:
                st.metric("P(Real)", f"{p_real * 100:.1f}%")

    if metrics and "accuracies" in metrics:
        st.subheader("Model comparison (test accuracy)")
        names = list(metrics["accuracies"].keys())
        accs = [metrics["accuracies"][n] for n in names]
        chart_df = pd.DataFrame({"Accuracy": accs}, index=names)
        st.bar_chart(chart_df, height=280)
        st.caption(
            f"Best model in last training run: **{metrics.get('best_model')}** "
            f"({metrics.get('best_accuracy', 0) * 100:.2f}% accuracy)."
        )
        for n in names:
            mark = " (saved)" if n == metrics.get("best_model") else ""
            st.write(f"• **{n}**: {metrics['accuracies'][n]:.4f}{mark}")


if __name__ == "__main__":
    main()
