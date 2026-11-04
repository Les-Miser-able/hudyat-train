from concurrent.futures import ThreadPoolExecutor
import csv

from hudyat_train.experiments import Registry
from hudyat_train.storage import write_json


def test_registry_updates_one_experiment_and_exports_metric_units(project):
    registry = Registry(project.root)
    directory = project.root / "runs" / "example"
    registry.save(directory, "training", {"status": "running", "category": "alphabet", "variant": "static"})
    registry.save(directory, "training", {"status": "complete", "category": "alphabet", "variant": "static"}, {"accuracy": 0.75, "macro_f1": 0.7})
    assert len(registry.records()) == 1
    with (project.root / "reports/experiments.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["accuracy_percent"]) == 75
    assert row["status"] == "complete"


def test_concurrent_writers_preserve_all_rows_and_csv(project):
    registry = Registry(project.root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: registry.save(project.root / f"run-{i}", "training", {"status": "complete"}, {"accuracy": 0.5}), range(4)))
    assert len(registry.records()) == 4
    with (project.root / "reports/experiments.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 4


def test_existing_runs_can_rebuild_history_without_tensorflow(project):
    folder = project.path("runs_dir") / "old-run"
    write_json(folder / "run.json", {"status": "complete", "category": "alphabet", "variant": "static", "protocol": "random"})
    write_json(folder / "metrics.json", {"accuracy": 0.9, "macro_f1": 0.88})
    registry = Registry(project.root)
    records = registry.reconcile(project.path("runs_dir"))
    assert len(records) == 1 and records[0]["metrics"]["accuracy"] == 0.9
    assert records[0]["kind"] == "training"
