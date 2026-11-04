import csv

from hudyat_train.catalog import Catalog
from hudyat_train.dataset import discover, scan
from .conftest import add_source


def test_accepts_labels_ids_and_nested_media(project):
    add_source(project, "alphabet/A/S1/video/first.MP4", b"first")
    add_source(project, "alphabet/letter_b/S2/image/photo.png", b"second")
    sources, issues, excluded = discover(project)
    assert not issues and not excluded
    assert [(s.sign_id, s.signer, s.kind) for s in sources] == [("letter_a", "S1", "video"), ("letter_b", "S2", "image")]


def test_static_excludes_moving_classes_and_images_cannot_teach_motion(project):
    add_source(project, "alphabet/J/S1/a.jpg", b"j image")
    add_source(project, "alphabet/Z/S1/a.mp4", b"z video")
    sources, issues, excluded = discover(project, variant="static")
    assert not sources and not issues and len(excluded) == 2
    sources, issues, excluded = discover(project, variant="sequence")
    assert len(sources) == 1 and sources[0].sign_id == "letter_z"
    assert len(excluded) == 1


def test_duplicate_content_never_crosses_class_or_signer(project):
    add_source(project, "alphabet/A/S1/one.mp4", b"same")
    add_source(project, "alphabet/A/S1/copy.mp4", b"same")
    add_source(project, "alphabet/A/S2/one.mp4", b"same")
    sources, issues, excluded = discover(project)
    assert len(sources) == 1 and len(issues) == 1 and len(excluded) == 1
    assert "different classes/signers" in issues[0]["reason"]


def test_flagged_metadata_and_temporary_files_are_excluded(project):
    add_source(project, "alphabet/A/S1/one.mp4", b"flagged")
    add_source(project, "alphabet/A/S1/one.tmp.mp4", b"unfinished")
    path = project.path("raw_dir").parent / "metadata.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file_path", "status", "orientation"])
        writer.writeheader()
        writer.writerow({"file_path": "dataset/raw/alphabet/A/S1/one.mp4", "status": "needs_rerecord", "orientation": "camera_raw_unmirrored"})
    sources, issues, excluded = discover(project)
    assert not sources and not issues and len(excluded) == 1


def test_coverage_includes_empty_classes_and_expected_signers(project):
    add_source(project)
    report = scan(project, "alphabet", "static", emit=lambda _: None)
    alphabet = report["coverage"][0]
    assert alphabet["classes_expected"] == 24
    assert alphabet["classes_present"] == 1
    assert alphabet["signs"][0]["counts"] == {"S1": 1, "S2": 0, "S3": 0}


def test_unknown_labels_and_public_signers_are_errors(project):
    add_source(project, "alphabet/unknown/S1/a.mp4", b"unknown")
    add_source(project, "alphabet/A/PUB/a.mp4", b"public")
    sources, issues, excluded = discover(project)
    assert not sources and len(issues) == 2


def test_fixed_vocabulary_matches_manuscript(project):
    catalog = Catalog(project.path("signs_file"))
    assert len(catalog.signs) == 108
    assert len({s.category for s in catalog.signs}) == 9
    assert {s.sign_id for s in catalog.signs if s.mode == "dynamic"} == {"letter_j", "letter_z"}
