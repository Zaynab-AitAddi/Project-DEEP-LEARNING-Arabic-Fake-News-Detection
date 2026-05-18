"""
Lance l'entraînement à partir d'un fichier YAML.

Usage:
  cd AFNDetection
  pip install -r requirements.txt
  pip install -e .
  python scripts/train.py --config configs/train_arabic.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fndarija.config import TrainConfig  # noqa: E402
from fndarija.training.run import train_from_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Chemin vers train_*.yaml")
    parser.add_argument("--max-samples", type=int, default=None, help="Surcharge (surtout arabe)")
    parser.add_argument("--epochs", type=float, default=None, help="Surcharge du nombre d'époques")
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(Path(args.config))
    if args.max_samples is not None:
        cfg.raw["max_samples"] = args.max_samples
    if args.epochs is not None:
        cfg.raw.setdefault("training", {})["num_train_epochs"] = float(args.epochs)
    train_from_config(cfg.raw, ROOT)


if __name__ == "__main__":
    main()
