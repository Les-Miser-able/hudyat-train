from dataclasses import replace
import io

import numpy as np
import pytest

from hudyat_train.inference import ModelInfo
from hudyat_train.storage import read_json, sha256, write_bytes, write_json
from hudyat_train.testing import TestSession as SessionLog, automated_test, summarize_attempts


def info(directory):
    return ModelInfo(directory, ["A", "B"], {"features": 128, "normalization": "hand-relative-sequence-v1", "confidence_threshold": 0.85},
                     "alphabet", "sequence", True, "fixture-model", {"status": "complete"})


def test_unlabelled_predictions_do_not_fabricate_accuracy():
    summary = summarize_attempts([{"expected": None, "predicted": "A", "confidence": 0.99, "accepted": True, "outcome": "unscored", "model_ms": 5}], ["A", "B"])
    assert summary["accuracy"] is None and summary["samples"] == 0
    assert summary["successful_attempt_rate"] is None and summary["unlabelled_attempts"] == 1


def test_rejections_and_invalid_inputs_remain_visible_in_success_rate():
    rows = [
        {"expected": "A", "predicted": "A", "accepted": True, "outcome": "correct", "model_ms": 4},
        {"expected": "A", "predicted": "A", "accepted": False, "outcome": "rejected", "model_ms": 6},
        {"expected": "A", "predicted": None, "accepted": False, "outcome": "invalid"},
    ]
    summary = summarize_attempts(rows, ["A", "B"])
    assert summary["accuracy"] == 1 and summary["samples"] == 2
    assert summary["successful_attempt_rate"] == pytest.approx(1 / 3)
    assert summary["invalid_attempts"] == 1 and summary["rejected_predictions"] == 1
    assert summary["untested_classes"] == ["B"]
    assert summary["latency"]["mean_model_ms"] == 5


def test_session_records_attempts_immediately_and_marks_failed_sessions(project, monkeypatch):
    monkeypatch.setattr("hudyat_train.reporting.result_report", lambda *a: None)
    session = SessionLog(project, info(project.root / "model"), "live_test")
    session.add({"predicted": "A", "confidence": 0.95, "model_ms": 5}, expected="A", mode="manual")
    assert read_json(session.directory / "attempts.json")[0]["outcome"] == "correct"
    assert (session.directory / "attempts.csv").is_file()
    session.close("failed", "camera disconnected")
    assert read_json(session.directory / "session.json")["status"] == "failed"
    assert read_json(session.directory / "summary.json")["samples"] == 1


def saved_run(project):
    run = project.path("runs_dir") / "alphabet" / "sequence" / "fixture"
    run.mkdir(parents=True)
    (run / "best.keras").write_bytes(b"fake checkpoint used only with the stub predictor")
    splits = {"train": [], "validation": [], "test": []}
    base = project.path("extracted_dir") / "sequence/alphabet"
    for label_i, label in enumerate(["A", "B"]):
        for name in splits:
            relative = f"letter_{label.lower()}/{name}.npy"
            sample = np.zeros((32, 128), np.float32)
            sample[:, 0], sample[:, 127] = label_i, 1
            data = io.BytesIO()
            np.save(data, sample, allow_pickle=False)
            write_bytes(base / relative, data.getvalue())
            splits[name].append({"digest": label + name, "label": label, "relative": f"alphabet/{label}/S1/{name}.mp4", "signer": "S1",
                                 "sample": relative, "sample_sha256": sha256(base / relative)})
    write_json(run / "classes.json", ["A", "B"])
    write_json(run / "preprocessing.json", info(run).preprocessing)
    write_json(run / "splits.json", splits)
    write_json(run / "dataset-index.json", {"settings": project.values})
    names = ["classes.json", "preprocessing.json", "splits.json", "dataset-index.json"]
    write_json(run / "run.json", {"status": "complete", "category": "alphabet", "variant": "sequence", "protocol": "random", "development": True,
                                  "best_model_sha256": sha256(run / "best.keras"), "artifact_sha256": {name: sha256(run / name) for name in names}})
    return run


def test_automated_testing_uses_only_saved_test_sources(project, monkeypatch):
    run = saved_run(project)
    calls = []
    class Stub:
        def __init__(self, info):
            pass
        def predict(self, sample):
            calls.append(sample)
            return {"predicted": "A" if sample[0, 0] == 0 else "B", "confidence": 0.95, "model_ms": 2}
    monkeypatch.setattr("hudyat_train.testing.Predictor", Stub)
    monkeypatch.setattr("hudyat_train.reporting.result_report", lambda *a: None)
    directory = automated_test(project, run)
    rows = read_json(directory / "attempts.json")
    assert len(calls) == 2 and all(row["source"].endswith("/test.mp4") for row in rows)
    assert read_json(directory / "summary.json")["accuracy"] == 1
    assert read_json(directory / "session.json")["status"] == "complete"


def test_changed_sample_is_rejected_before_model_loading(project, monkeypatch):
    run = saved_run(project)
    (project.path("extracted_dir") / "sequence/alphabet/letter_a/test.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        automated_test(project, run)


def test_automated_testing_refuses_train_test_overlap(project):
    run = saved_run(project)
    splits = read_json(run / "splits.json")
    splits["test"][0]["digest"] = splits["train"][0]["digest"]
    write_json(run / "splits.json", splits)
    state = read_json(run / "run.json")
    state["artifact_sha256"]["splits.json"] = sha256(run / "splits.json")
    write_json(run / "run.json", state)
    with pytest.raises(ValueError, match="overlap"):
        automated_test(project, run)
