"""Export train/val/test splits to Parquet (optional inspection). Arabic config only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fndarija.config import TrainConfig
from fndarija.data.loaders import load_arabic_hub, load_arabic_local, stratified_split_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = TrainConfig.from_yaml(Path(args.config)).raw
    language = cfg.get("language", "arabic")
    if language != "arabic":
        raise SystemExit(f"Only arabic configs are supported (got {language!r}).")
    seed = int(cfg.get("seed", 42))
    out = Path(cfg.get("data_paths", {}).get("processed_dir", "data/processed"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    ds_cfg = cfg["dataset"]
    local_path = ds_cfg.get("local_path")
    lp = Path(local_path).expanduser() if local_path else None
    if lp is not None and not lp.is_absolute():
        lp = ROOT / lp
    if lp is not None and lp.exists():
        df = load_arabic_local(
            lp,
            text_mode=ds_cfg.get("text_mode", "title_plus_text"),
            max_samples=cfg.get("max_samples"),
            seed=seed,
        )
    else:
        df = load_arabic_hub(
            hub_id=ds_cfg["hub_id"],
            split=ds_cfg.get("split", "train"),
            text_mode=ds_cfg.get("text_mode", "title_plus_text"),
            max_samples=cfg.get("max_samples"),
            seed=seed,
        )
    train_df, val_df, test_df = stratified_split_df(
        df,
        "label",
        float(cfg.get("val_ratio", 0.1)),
        float(cfg.get("test_ratio", 0.1)),
        seed,
    )
    prefix = "arabic"
    train_df.to_parquet(out / f"{prefix}_train.parquet", index=False)
    val_df.to_parquet(out / f"{prefix}_val.parquet", index=False)
    test_df.to_parquet(out / f"{prefix}_test.parquet", index=False)
    print(f"Écrit: {out}/{prefix}_*.parquet  (lignes train={len(train_df)})")


if __name__ == "__main__":
    main()
