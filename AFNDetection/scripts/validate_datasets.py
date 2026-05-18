"""
Check that the Arabic (Hugging Face) training config is reachable.

Usage (from project root):
  python scripts/validate_datasets.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402


def _load_yaml(name: str) -> dict:
    p = ROOT / "configs" / name
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML: {p}")
    return raw


def validate_arabic() -> list[str]:
    issues: list[str] = []
    cfg = _load_yaml("train_arabic.yaml")
    hub_id = cfg.get("dataset", {}).get("hub_id")
    if not hub_id:
        issues.append("train_arabic.yaml: missing dataset.hub_id")
        return issues
    try:
        from datasets import load_dataset

        ds = load_dataset(hub_id, split=cfg.get("dataset", {}).get("split", "train"), streaming=True)
        row = next(iter(ds))
        if row.get("label") is None and "label" not in row:
            issues.append(f"HF {hub_id}: unexpected schema (no label): keys={list(row.keys())}")
    except Exception as e:
        issues.append(f"HF dataset '{hub_id}' not reachable or schema mismatch: {e}")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate AFNDetection Arabic dataset config")
    parser.parse_args()
    raw_issues = validate_arabic()
    if not raw_issues:
        print("OK — Arabic Hub config looks reachable.")
        raise SystemExit(0)

    print("Issues:\n")
    for m in raw_issues:
        print(f"[Arabic] {m}")
        print()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
