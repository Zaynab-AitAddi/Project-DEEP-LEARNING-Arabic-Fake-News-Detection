from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
