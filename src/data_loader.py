"""Load or download fake vs. real news data; handle missing values."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_URL = (
    "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/"
    "fake_or_real_news.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV = DATA_DIR / "fake_or_real_news.csv"


def download_dataset(
    url: str = DEFAULT_URL,
    dest: Path | None = None,
    timeout: int = 600,
) -> Path:
    """Download the combined fake/real news CSV if missing."""
    dest = dest or DEFAULT_CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "fake-news-detection/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_news_dataframe(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load dataset with columns that include article text and a binary label.

    Supports:
    - lutzhamel fake_or_real_news.csv: id, title, text, label (FAKE|REAL)
    - Generic: 'text' + 'label', or 'title'/'text' + 'label'
    """
    path = csv_path or DEFAULT_CSV
    if not path.exists():
        download_dataset(dest=path)

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    # Normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    title_col = cols.get("title")
    text_col = cols.get("text")
    label_col = cols.get("label")

    if text_col is None:
        raise ValueError(
            f"Expected a 'text' column in {path}. Found: {list(df.columns)}"
        )
    if label_col is None:
        raise ValueError(
            f"Expected a 'label' column in {path}. Found: {list(df.columns)}"
        )

    # Missing titles / text
    if title_col is not None:
        df[title_col] = df[title_col].fillna("").astype(str)
    df[text_col] = df[text_col].fillna("").astype(str)

    if title_col is not None:
        combined = (df[title_col].str.strip() + " " + df[text_col].str.strip()).str.strip()
    else:
        combined = df[text_col].str.strip()

    df = df.assign(_article_text=combined)

    # Drop rows with empty article text
    df = df[df["_article_text"].str.len() > 0]

    # Normalize labels to FAKE (0) / REAL (1)
    raw = df[label_col].astype(str).str.strip().str.upper()
    label_map = {
        "FAKE": 0,
        "REAL": 1,
        "0": 0,
        "1": 1,
        0: 0,
        1: 1,
    }
    y = raw.map(label_map)
    before = len(df)
    df = df.assign(_label=y)
    df = df[df["_label"].notna()]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with unknown or missing labels.")

    df = df.drop_duplicates(subset=["_article_text"], keep="first")
    return df.reset_index(drop=True)


def get_X_y(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    X = df["_article_text"].tolist()
    y = df["_label"].astype(int).tolist()
    return X, y


if __name__ == "__main__":
    p = download_dataset()
    print(f"Dataset ready at {p}")
    d = load_news_dataframe(p)
    print(d[["_article_text", "_label"]].head())
    print("Rows:", len(d))
