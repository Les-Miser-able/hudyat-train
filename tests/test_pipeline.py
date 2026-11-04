import numpy as np
import pytest

from hudyat_train.augmentation import augment
from hudyat_train.evaluation import classification_report
from hudyat_train.extraction import extract
from hudyat_train.features import mirror_frames, prepare
from hudyat_train.jobs import Cancelled, Job
from hudyat_train.splitting import loso_splits, random_split
from hudyat_train.storage import read_json
from hudyat_train.training import TrainOptions, load_dataset
from .conftest import add_source


def test_extraction_reuses_detections_and_tracks_current_files(project, detections, monkeypatch):
    calls = []
    monkeypatch.setattr("hudyat_train.extraction.detect_media", lambda *a: calls.append(a) or detections)
    add_source(project)
    index = extract(project, "alphabet", "sequence")
    assert index["items"][0]["status"] == "accepted"
    assert index["input_shape"] == [32, 128]
    extract(project, "alphabet", "static")
    assert len(calls) == 1  # the detector cache is shared, but the derived samples are separate
    assert (project.path("raw_dir") / "alphabet/A/S1/take.mp4").read_bytes() == b"unique recording"


def test_quality_is_checked_in_window_and_rejected_sources_keep_reason(project, detections, monkeypatch):
    data = dict(detections)
    data["raw"] = detections["raw"].copy()
    data["raw"][10:20] = 0
    monkeypatch.setattr("hudyat_train.extraction.detect_media", lambda *a: data)
    add_source(project)
    index = extract(project, "alphabet", "sequence")
    item = index["items"][0]
    assert item["status"] == "rejected" and "below" in item["reason"]
    assert item["quality"]["hand_rate"] == 16 / 32


def test_source_change_invalidates_training_index(project, detections, monkeypatch):
    monkeypatch.setattr("hudyat_train.extraction.detect_media", lambda *a: detections)
    path = add_source(project)
    extract(project, "alphabet", "sequence")
    path.write_bytes(b"new recording")
    with pytest.raises(ValueError, match="changed since extraction"):
        load_dataset(project, TrainOptions("alphabet", development=True))


def test_cancellation_leaves_previous_index_intact(project, detections, monkeypatch):
    monkeypatch.setattr("hudyat_train.extraction.detect_media", lambda *a: detections)
    add_source(project)
    index = extract(project, "alphabet", "sequence")
    cancel = project.root / "cancel"
    cancel.write_text("stop")
    with pytest.raises(Cancelled):
        extract(project, "alphabet", "sequence", Job(cancel))
    assert read_json(project.path("extracted_dir") / "sequence/alphabet/index.json") == index


def items():
    return [{"sign_id": label, "signer": signer, "digest": f"{label}-{signer}-{take}"}
            for label in ("a", "b") for signer in ("S1", "S2", "S3") for take in range(15)]


def test_random_split_is_reproducible_stratified_and_source_disjoint():
    records = items()
    split = random_split(records)
    assert split == random_split(records)
    assert len(split["train"]) == 62 and len(split["validation"]) == 14 and len(split["test"]) == 14
    assert set(split["train"]).isdisjoint(split["test"])
    assert sorted(sum(split.values(), [])) == list(range(len(records)))


def test_loso_keeps_training_validation_and_test_signers_independent():
    records = items()
    folds = loso_splits(records)
    assert len(folds) == 3
    assert {fold["test_signer"] for fold in folds} == {"S1", "S2", "S3"}
    for fold in folds:
        groups = [{records[i]["signer"] for i in fold["split"][name]} for name in ("train", "validation", "test")]
        assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))


def test_one_signer_cannot_claim_loso():
    with pytest.raises(ValueError, match="at least 3 signers"):
        loso_splits([{"sign_id": "a", "signer": "S1", "digest": "a"}])


def test_augmentation_is_deterministic_keeps_binary_flags_and_missing_hands_zero(detections):
    sample, _, _ = prepare(detections["raw"], detections["locations"], 1280, 720, 30, "alphabet", "sequence")
    original = sample.copy()
    a = augment(sample, np.random.default_rng(42), True)
    b = augment(sample, np.random.default_rng(42), True)
    assert np.array_equal(a, b) and np.array_equal(sample, original)
    assert set(a[:, 126:128].ravel()) <= {0, 1}
    for slot in range(2):
        assert not a[a[:, 126 + slot] == 0, slot * 63:(slot + 1) * 63].any()


def test_static_augmentation_keeps_a_held_pose(detections):
    sample, _, _ = prepare(detections["raw"], detections["locations"], 1280, 720, 30, "alphabet", "static")
    augmented = augment(sample, np.random.default_rng(42), True, static=True)
    assert np.all(augmented == augmented[0])


@pytest.mark.parametrize("category,variant", [("alphabet", "sequence"), ("alphabet", "static"),
                                             ("family", "sequence"), ("family", "static")])
def test_requested_mirrored_augmentation_always_reflects_the_same_perturbation(detections, category, variant):
    sample, _, _ = prepare(detections["raw"], detections["locations"], 1280, 720, 30, category, variant)
    static = variant == "static"
    plain = augment(sample, np.random.default_rng(42), static=static)
    mirrored = augment(sample, np.random.default_rng(42), mirror=True, static=static)
    np.testing.assert_array_equal(mirrored, mirror_frames(plain))


def test_metrics_keep_confidence_and_classification_accuracy_separate():
    report = classification_report([0, 0, 1, 1], [0, 1, 1, 1], ["A", "B"])
    assert report["accuracy"] == 0.75
    assert report["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert report["benchmark"]["signs_below_70_recall"] == ["A"]
    assert not report["benchmark"]["classification_pass"]


def test_study_run_does_not_silently_accept_incomplete_data(project, detections, monkeypatch):
    monkeypatch.setattr("hudyat_train.extraction.detect_media", lambda *a: detections)
    add_source(project)
    extract(project, "alphabet", "sequence")
    with pytest.raises(ValueError, match="explicitly choose a development run"):
        load_dataset(project, TrainOptions("alphabet"))
