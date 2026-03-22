# Fake News Detection System with Dashboard

Python project that classifies news articles as **Fake** or **Real** using NLP preprocessing, **TF–IDF** vectorization, and three scikit-learn models: **Logistic Regression**, **PassiveAggressiveClassifier**, and **Random Forest**. The best model (by hold-out accuracy) is saved and used in a **Streamlit** dashboard.

## Project layout

```
fake-news-detection/
├── data/              # CSV downloaded here on first training run
├── src/
│   ├── data_loader.py # Download + load dataset, handle missing values
│   ├── preprocess.py  # Cleaning + English stopword removal
│   ├── train.py       # Train models, compare, save best
│   └── predict.py     # Load artifacts and predict
├── models/            # Saved vectorizer, classifier, metrics JSON
├── app.py             # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Dataset

By default, training downloads [lutzhamel/fake-news `fake_or_real_news.csv`](https://github.com/lutzhamel/fake-news/blob/master/data/fake_or_real_news.csv) into `data/fake_or_real_news.csv` (combined fake and real articles with `title`, `text`, and `label`).

You can place your own CSV in `data/` as `fake_or_real_news.csv` with at least:

- `text` — article body (required)
- `label` — `FAKE` / `REAL` (or `0` / `1` for fake/real)
- `title` — optional; concatenated with `text` when present

Rows with empty `text`, unknown labels, or duplicate combined text are dropped.

## Setup

```bash
cd fake-news-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

NLTK English stopwords are downloaded automatically on first use into `.nltk_data/` inside this project (no manual `nltk.download` step).

## Train models

```bash
python -m src.train
```

This preprocesses text, fits TF–IDF, trains all three models, prints accuracy and classification reports, and writes:

- `models/tfidf_vectorizer.joblib`
- `models/best_classifier.joblib`
- `models/training_metrics.json`

## Dashboard

```bash
streamlit run app.py
```

The app provides a text box, prediction label, class probabilities (with an approximate score for Passive Aggressive when needed), and a bar chart of each model’s test accuracy from the last training run.

## Requirements summary

- Python 3.10+
- NLP preprocessing and stopword removal (NLTK)
- TF–IDF vectorization
- Model training, comparison, accuracy evaluation, best-model persistence
- Streamlit UI with prediction and accuracy chart
