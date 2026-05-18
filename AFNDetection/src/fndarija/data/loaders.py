from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from fndarija.data.preprocess import compose_document


def _normalize_label(v: Any) -> int | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        if v in (0, 1):
            return int(v)
        return None
    s = str(v).strip().lower()
    if s in {"0", "fake", "false", "fausse", "mentira"}:
        return 0
    if s in {"1", "real", "true", "vrai", "authentic", "authentique"}:
        return 1
    return None


def load_arabic_local(
    path: str | Path,
    text_mode: str,
    max_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    """Load curated Arabic rows from CSV / Parquet (columns: title?, text?, label)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    suf = path.suffix.lower()
    if suf == ".csv":
        raw = pd.read_csv(path, encoding="utf-8")
    elif suf == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path} (use .csv or .parquet)")
    has_text = "text" in raw.columns
    has_title = "title" in raw.columns
    if not has_text and not has_title:
        raise ValueError("Local dataset needs at least one of: title, text")
    if "label" not in raw.columns:
        raise ValueError("Local dataset needs column: label (0=fake, 1=real)")
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(raw))
    rng.shuffle(idxs)
    mode = _map_text_mode(text_mode)
    for i in idxs:
        r = raw.iloc[int(i)]
        label = _normalize_label(r.get("label"))
        if label is None:
            continue
        title = r.get("title") if has_title else None
        body = r.get("text") if has_text else None
        text = compose_document(title, body, mode)
        if not text:
            continue
        rows.append({"text": text, "label": label})
        if max_samples is not None and len(rows) >= max_samples:
            break
    return pd.DataFrame(rows)


def load_arabic_hub(
    hub_id: str,
    split: str,
    text_mode: str,
    max_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    ds = load_dataset(hub_id, split=split)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(ds))
    rng.shuffle(indices)
    for i in indices:
        ex = ds[int(i)]
        label = _normalize_label(ex.get("label"))
        if label is None:
            continue
        text = compose_document(ex.get("title"), ex.get("text"), _map_text_mode(text_mode))
        if not text:
            continue
        rows.append({"text": text, "label": label})
        if max_samples is not None and len(rows) >= max_samples:
            break
    return pd.DataFrame(rows)


def _map_text_mode(mode: str) -> str:
    if mode == "title_plus_text":
        return "title_plus_text"
    if mode == "title_only":
        return "title_only"
    if mode == "text_only":
        return "text_only"
    raise ValueError(f"text_mode inconnu: {mode}")


def stratified_split_df(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio doit être < 1.0")
    strat = df[label_col].astype(int)
    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=strat,
        random_state=seed,
    )
    val_size = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df[label_col],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def dataframes_to_hf(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> DatasetDict:
    return DatasetDict(
        train=Dataset.from_pandas(train, preserve_index=False),
        validation=Dataset.from_pandas(val, preserve_index=False),
        test=Dataset.from_pandas(test, preserve_index=False),
    )
