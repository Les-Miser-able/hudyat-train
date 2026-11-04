from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .storage import read_json, within

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("sequence", "static")


@dataclass(frozen=True)
class Settings:
    root: Path
    values: dict

    def path(self, key: str) -> Path:
        return within(self.root, self.values[key])

    def get(self, key: str):
        return self.values[key]


def load_settings(root: Path = ROOT) -> Settings:
    root = root.resolve()
    values = read_json(root / "project.json")
    for key in ("raw_dir", "extracted_dir", "cache_dir", "runs_dir", "exports_dir", "hand_model", "face_model", "signs_file"):
        within(root, values[key])
    for key in ("detection_confidence", "min_hand_rate", "min_both_hands_rate"):
        if not 0 < float(values[key]) <= 1:
            raise ValueError(f"{key} must be between 0 and 1.")
    return Settings(root, values)
