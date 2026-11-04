"""Synthetic fixtures for checking Python against the real HUDYAT runtime."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np

from .features import (auto_hand, location_features, mirror_frames, normalize_frame,
                       normalize_sequence, resample_legacy, resample_sign, sign_window)
from .storage import write_json


def write_fixture(path: Path) -> Path:
    rng = np.random.default_rng(31415)
    cases = []
    for width, height in [(1280, 720), (640, 480), (720, 1280)]:
        raw = np.zeros((11, 128), np.float32)
        for t in range(len(raw)):
            for slot in range(2):
                if (slot == 0 and t in (0, 3, 4, 10)) or (slot == 1 and t in (0, 7, 10)):
                    continue
                points = rng.uniform(-0.04, 0.04, (21, 3)) + [0.3 + slot * 0.3 + t * 0.015, 0.45, 0]
                points[9] = points[0] + [0.02, -0.06, 0.01]
                raw[t, slot * 63:(slot + 1) * 63] = points.ravel()
                raw[t, 126 + slot] = 1
        face = [{"x": 0.5, "y": 0.3} for _ in range(153)]
        face[10], face[152] = {"x": 0.49, "y": 0.1}, {"x": 0.51, "y": 0.5}
        faces = [None if t in (3, 8) else face for t in range(len(raw))]
        normalized = np.asarray([normalize_frame(f, width, height) for f in raw])
        located = []
        for f, norm, points in zip(raw, normalized, faces):
            location = location_features(f, [SimpleNamespace(**p) for p in points] if points else None, width, height)
            located.append(np.r_[norm, location].astype(np.float32))
        located = np.asarray(located)
        sequence = normalize_sequence(raw, width, height)
        present = np.any(raw[:, 126:128] > 0.5, axis=1).tolist()
        cases.append({"width": width, "height": height, "raw": raw.tolist(), "faces": faces,
                      "normalized": normalized.tolist(), "sequence": sequence.tolist(), "located": located.tolist(),
                      "legacy": resample_legacy(sequence).tolist(), "sign": resample_sign(located).tolist(),
                      "mirror": mirror_frames(located).tolist(), "auto_hand": auto_hand(raw, width, height),
                      "window": list(sign_window(present, 25))})
    write_json(path, {"cases": cases})
    return path.resolve()
