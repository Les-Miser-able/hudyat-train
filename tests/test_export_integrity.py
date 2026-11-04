import pytest

from hudyat_train.exporting import export_run
from hudyat_train.storage import sha256, write_json


def test_loso_and_incomplete_runs_cannot_be_exported(project, tmp_path):
    run = tmp_path / "run"
    write_json(run / "run.json", {"status": "complete", "protocol": "loso"})
    with pytest.raises(ValueError, match="LOSO runs are evaluation"):
        export_run(project, run)


def test_development_export_is_explicit(project, tmp_path):
    run = tmp_path / "run"
    write_json(run / "run.json", {"status": "complete", "protocol": "random", "development": True})
    with pytest.raises(ValueError, match="--allow-development"):
        export_run(project, run)


def test_changed_class_order_is_rejected_before_tensorflow_load(project, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "best.keras").write_bytes(b"checkpoint fixture")
    names = ("classes.json", "class-map.json", "preprocessing.json", "training_config.json", "dataset-index.json", "splits.json", "metrics.json")
    for name in names:
        write_json(run / name, ["A", "B"] if name == "classes.json" else {})
    state = {"status": "complete", "protocol": "random", "development": False,
             "best_model_sha256": sha256(run / "best.keras"), "artifact_sha256": {name: sha256(run / name) for name in names}}
    write_json(run / "run.json", state)
    write_json(run / "classes.json", ["B", "A"])
    with pytest.raises(ValueError, match="classes.json changed after evaluation"):
        export_run(project, run)
