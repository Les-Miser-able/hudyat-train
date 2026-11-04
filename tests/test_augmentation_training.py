import csv
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hudyat_train import cli, training
from hudyat_train.features import mirror_frames, prepare
from hudyat_train.jobs import Job
from hudyat_train.storage import read_json
from hudyat_train.training import TrainOptions


@pytest.mark.parametrize("copies", [0, 2, 5])
def test_cli_passes_the_selected_augmentation_count(project, monkeypatch, copies):
    received = []
    monkeypatch.setattr(cli.sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))
    monkeypatch.setattr(training, "train", lambda settings, options, job: received.append(options) or project.root / "run")
    assert cli.main(["--root", str(project.root), "train", "--category", "alphabet", "--development",
                     "--augmentation-copies", str(copies)]) == 0
    assert received[0].augmentation_copies == copies
    assert received[0].mirroring == "all"
    assert received[0].development


@pytest.mark.parametrize("mirroring", ["all", "approved", "off"])
def test_cli_passes_mirroring_policy(project, monkeypatch, mirroring):
    received = []
    monkeypatch.setattr(cli.sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))
    monkeypatch.setattr(training, "train", lambda settings, options, job: received.append(options) or project.root / "run")
    assert cli.main(["--root", str(project.root), "train", "--category", "alphabet", "--mirroring", mirroring]) == 0
    assert received[0].mirroring == mirroring


@pytest.fixture
def captured_training(monkeypatch):
    class Dataset:
        def __init__(self, values):
            self.x, self.y = values

        def shuffle(self, *args, **kwargs):
            return self

        def batch(self, *args):
            return self

        def with_options(self, *args):
            return self

    class CapturedTraining(Exception):
        pass

    fitted = {}

    class Model:
        def compile(self, **kwargs):
            pass

        def fit(self, data, validation_data, **kwargs):
            fitted.update(training=data, validation=validation_data)
            raise CapturedTraining()

    callbacks = SimpleNamespace(Callback=object, **{name: lambda *args, **kwargs: None for name in
                                ("ModelCheckpoint", "EarlyStopping", "ReduceLROnPlateau")})
    keras = SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None),
                            utils=SimpleNamespace(set_random_seed=lambda seed: None), callbacks=callbacks,
                            optimizers=SimpleNamespace(Adam=lambda rate: None))
    fake_tf = SimpleNamespace(__version__="2.19.1", keras=keras, data=SimpleNamespace(
        Dataset=SimpleNamespace(from_tensor_slices=Dataset), Options=lambda: SimpleNamespace(threading=SimpleNamespace())))
    monkeypatch.setattr(training, "tensorflow", lambda: fake_tf)
    monkeypatch.setattr(training, "build_model", lambda *args: Model())
    monkeypatch.setattr(training, "runtime_details", lambda: {})
    return fitted, CapturedTraining


@pytest.mark.parametrize("copies", [0, 2, 5])
def test_training_augments_only_training_sources_and_records_counts(tmp_path, monkeypatch, copies, captured_training):
    fitted, CapturedTraining = captured_training
    augmented_sources = []

    def augment(sample, *args):
        augmented_sources.append(float(sample[0, 0]))
        return sample + 100

    monkeypatch.setattr(training, "augment", augment)
    samples = np.stack([np.full((32, 128), i, np.float32) for i in range(6)])
    originals = samples.copy()
    labels = np.array([0, 1, 0, 1, 0, 1])
    items = [{"digest": str(i), "relative": f"take-{i}.mp4", "signer": "S1", "label": "AB"[i % 2],
              "sign_id": "ab"[i % 2], "mirror_ok": False} for i in range(6)]
    split = {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}
    manifest = {"detectors": {}, "input_shape": [32, 128], "normalization": "hand-relative-sequence-v1",
                "settings": {"min_hand_rate": 0.8}}
    run = tmp_path / "run"
    with pytest.raises(CapturedTraining):
        training._fit(run, TrainOptions("alphabet", augmentation_copies=copies, development=True, mirroring="off"),
                      manifest, items, samples, labels, ["A", "B"], split, Job())
    assert len(fitted["training"].x) == 2 * (1 + copies)
    assert augmented_sources == [0.0] * copies + [1.0] * copies
    assert np.array_equal(fitted["validation"].x, originals[[2, 3]])
    assert np.array_equal(samples, originals)
    with (run / "augmentation.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["source"] for row in rows} == {"take-0.mp4", "take-1.mp4"}
    assert sum(int(row["augmented_copies"]) for row in rows) == 2 * copies
    assert read_json(run / "run.json")["generated_training_samples"] == 2 * copies


@pytest.mark.parametrize("category,variant", [("alphabet", "sequence"), ("alphabet", "static"),
                                             ("family", "sequence"), ("family", "static")])
@pytest.mark.parametrize("mirroring,copies", [("all", 5), ("approved", 5), ("off", 5), ("all", 1), ("all", 0)])
def test_training_guarantees_opposite_hand_examples_and_records_them(tmp_path, detections, captured_training,
                                                                   category, variant, mirroring, copies):
    fitted, CapturedTraining = captured_training
    sample, normalization, _ = prepare(detections["raw"], detections["locations"], 1280, 720, 30, category, variant)
    samples = np.stack([sample.copy() for _ in range(6)])
    originals = samples.copy()
    labels = np.array([0, 1, 0, 1, 0, 1])
    items = [{"digest": str(i), "relative": f"take-{i}.mp4", "signer": "S1", "label": "AB"[i % 2],
              "sign_id": "ab"[i % 2], "mirror_ok": bool(i % 2)} for i in range(6)]
    split = {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}
    manifest = {"detectors": {}, "input_shape": list(sample.shape), "normalization": normalization,
                "settings": {"min_hand_rate": 0.8}}
    options = TrainOptions(category, variant, augmentation_copies=copies, development=True, mirroring=mirroring)
    run = tmp_path / "run"
    with pytest.raises(CapturedTraining):
        training._fit(run, options, manifest, items, samples, labels, ["A", "B"], split, Job())
    data = fitted["training"]
    expected_counts = []
    for source in range(2):
        group = data.x[source * (copies + 1):(source + 1) * (copies + 1)]
        allowed = mirroring == "all" or (mirroring == "approved" and items[source]["mirror_ok"])
        expected = (copies + 1) // 2 if allowed else 0
        expected_counts.append(expected)
        assert np.array_equal(group[0], originals[source])
        if expected:
            assert np.array_equal(group[1], mirror_frames(originals[source]))
        assert sum(bool(frame[:, 126].any() and not frame[:, 127].any()) for frame in group) == expected
        assert data.y[source * (copies + 1):(source + 1) * (copies + 1)].tolist() == [source] * (copies + 1)
        for generated in group:
            for slot in range(2):
                missing = generated[:, 126 + slot] == 0
                assert not generated[missing, slot * 63:(slot + 1) * 63].any()
                if category == "family":
                    assert not generated[missing, 128 + slot * 6:134 + slot * 6].any()
            if variant == "static":
                assert np.all(generated == generated[0])
    assert np.array_equal(fitted["validation"].x, originals[[2, 3]])
    assert np.array_equal(samples, originals)
    with (run / "augmentation.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["mirrored_copies"]) for row in rows] == expected_counts
    assert [row["mirror_approved"] for row in rows] == ["False", "True"]
    assert {row["source"] for row in rows} == {"take-0.mp4", "take-1.mp4"}
    assert read_json(run / "run.json")["mirrored_training_samples"] == sum(expected_counts)
    assert read_json(run / "training_config.json")["mirroring"] == mirroring
    assert read_json(run / "preprocessing.json")["augmentation"]["mirroring"] == mirroring
    events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    event = next(event for event in events if event["type"] == "augmentation")
    assert event["mirrored_samples"] == sum(expected_counts)
    assert event["total_training_samples"] == 2 * (copies + 1)


def test_invalid_mirroring_policy_is_rejected_before_loading_data(project):
    with pytest.raises(ValueError, match="Mirroring must be"):
        training.train(project, TrainOptions("alphabet", mirroring="invalid"))
