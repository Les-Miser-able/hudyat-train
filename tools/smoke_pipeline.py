"""Exercise training/export on synthetic data in tmp/, never the study dataset.

Run with a Python environment containing TensorFlow, NumPy, Pillow, and Matplotlib.
These scores measure no FSL recognition capability. Real extraction is checked separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from hudyat_train.config import ROOT, load_settings
from hudyat_train.extraction import extract
from hudyat_train.exporting import export_run
from hudyat_train.storage import read_json, write_json
from hudyat_train.training import TrainOptions, train


def synthetic(source, settings, job):
    frames = np.zeros((32, 128), np.float32)
    locations = np.zeros((32, 13), np.float32)
    locations[:, 12] = 1
    code = int(source.digest[:6], 16)
    for t in range(3, 29):
        points = np.zeros((21, 3), np.float32)
        points[:, 0] = 0.55 + t * 0.002 + np.arange(21) * 0.003 + code % 10 * 0.001
        points[:, 1] = 0.6 - np.arange(21) * 0.006
        points[:, 2] = np.arange(21) * -0.001
        frames[t, 63:126], frames[t, 127] = points.ravel(), 1
        locations[t, 6:12] = [0.1 + t * 0.01, 0.2, 0.3, 0.4, 0.5, 0.6]
    return {"raw": frames, "locations": locations, "width": 1280, "height": 720, "fps": 30.0, "collisions": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "tmp" / "synthetic-smoke")
    args = parser.parse_args()
    project = args.work.resolve()
    project.mkdir(parents=True, exist_ok=True)
    if (project / "smoke-results.json").exists():
        raise ValueError("Choose a fresh --work directory for a new synthetic smoke run.")
    values = read_json(ROOT / "project.json")
    values["expected_signers"], values["takes_per_signer"] = ["S1", "S2", "S3"], 3
    write_json(project / "project.json", values)
    shutil.copyfile(ROOT / "signs.csv", project / "signs.csv")
    (project / "models").mkdir(exist_ok=True)
    for name in ("hand_landmarker.task", "face_landmarker.task"):
        shutil.copyfile(ROOT / "models" / name, project / "models" / name)
    settings = load_settings(project)
    for category, labels in [("alphabet", ["A", "B"]), ("family", ["Father", "Mother"])]:
        for label_i, label in enumerate(labels):
            for signer in range(1, 4):
                for take in range(3):
                    path = settings.path("raw_dir") / category / label / f"S{signer}" / f"{take}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (64, 64), (label_i * 80 + 20, signer * 50, take * 60)).save(path)
    results = []
    for category in ("alphabet", "family"):
        for variant in ("sequence", "static"):
            with patch("hudyat_train.extraction.detect_media", synthetic):
                extract(settings, category, variant)
            run = train(settings, TrainOptions(category, variant, epochs=1, development=True))
            output = export_run(settings, run, allow_development=True)
            results.append({"category": category, "variant": variant, "run": str(run), "bundle": str(output)})
            write_json(project / "smoke-results.partial.json", results)
    write_json(project / "smoke-results.json", {"synthetic_only": True, "results": results})
    print("Synthetic training/export smoke complete:", project)


if __name__ == "__main__":
    main()
