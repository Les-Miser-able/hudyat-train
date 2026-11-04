from pathlib import Path
import shutil

import numpy as np
import pytest

from hudyat_train.config import ROOT, load_settings
from hudyat_train.storage import read_json, write_json


@pytest.fixture
def project(tmp_path):
    values = read_json(ROOT / "project.json")
    values["expected_signers"] = ["S1", "S2", "S3"]
    values["takes_per_signer"] = 1
    write_json(tmp_path / "project.json", values)
    shutil.copyfile(ROOT / "signs.csv", tmp_path / "signs.csv")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "hand_landmarker.task").write_bytes(b"hand detector fixture")
    (tmp_path / "models" / "face_landmarker.task").write_bytes(b"face detector fixture")
    return load_settings(tmp_path)


@pytest.fixture
def detections():
    raw = np.zeros((32, 128), np.float32)
    for t in range(3, 29):
        points = np.zeros((21, 3), np.float32)
        points[:, 0] = 0.65 + t * 0.002 + np.arange(21) * 0.002
        points[:, 1] = 0.65 - np.arange(21) * 0.006
        points[:, 2] = np.arange(21) * -0.001
        raw[t, 63:126] = points.ravel()
        raw[t, 127] = 1
    locations = np.zeros((32, 13), np.float32)
    locations[:, 12] = 1
    locations[3:29, 6:12] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    return {"raw": raw, "locations": locations, "width": 1280, "height": 720, "fps": 30.0, "collisions": 0}


def add_source(project, folder="alphabet/A/S1/take.mp4", content=b"unique recording"):
    path = project.path("raw_dir") / folder
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path
