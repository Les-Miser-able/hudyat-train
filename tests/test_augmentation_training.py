import csv
import io
from types import SimpleNamespace

import numpy as np
import pytest

from hudyat_train import cli, training
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
    assert received[0].development


@pytest.mark.parametrize("copies", [0, 2, 5])
def test_training_augments_only_training_sources_and_records_counts(tmp_path, monkeypatch, copies):
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
        training._fit(run, TrainOptions("alphabet", augmentation_copies=copies, development=True),
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
