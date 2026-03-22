"""NLP preprocessing: cleaning and English stopword removal."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import nltk
from nltk.corpus import stopwords

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_NLTK_DATA_DIR = _PROJECT_ROOT / ".nltk_data"


def _ensure_nltk_stopwords() -> None:
    _NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if str(_NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(_NLTK_DATA_DIR))
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download(
            "stopwords",
            download_dir=str(_NLTK_DATA_DIR),
            quiet=True,
        )


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    _ensure_nltk_stopwords()
    return frozenset(stopwords.words("english"))


def preprocess_text(text: str | float | None) -> str:
    """
    Lowercase, remove non-letters, drop stopwords and very short tokens.
    """
    if text is None or (isinstance(text, float) and str(text) == "nan"):
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    stops = _english_stopwords()
    out = [t for t in tokens if t not in stops and len(t) > 1]
    return " ".join(out)


def preprocess_series(raw_texts: list[str]) -> list[str]:
    return [preprocess_text(t) for t in raw_texts]
