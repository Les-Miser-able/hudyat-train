"""A local experiment history, shared safely by desktop and CLI processes."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import platform
from pathlib import Path
import sqlite3

from .storage import read_json, write_bytes


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def runtime_details():
    packages = {}
    for name in ("tensorflow", "keras", "numpy", "mediapipe", "opencv-contrib-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "os": platform.system() + " " + platform.release(),
            "architecture": platform.machine(), "packages": packages}


def write_csv(path: Path, rows: list[dict], fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_bytes(path, stream.getvalue().encode("utf-8-sig"))


def append_event(path: Path, event: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": utc_now(), **event}, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


class Registry:
    def __init__(self, project: Path):
        self.project = project.resolve()
        self.directory = self.project / "reports"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "experiments.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, body TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.database, timeout=20)

    def save(self, directory: Path, kind: str, state: dict, metrics=None, parameters=None, export=True):
        directory = directory.resolve()
        identifier = hashlib.sha256(str(directory).casefold().encode()).hexdigest()[:20]
        record = {**state, "id": identifier, "path": str(directory), "kind": kind,
                  "metrics": metrics or {}, "parameters": parameters or {}, "updated_at": utc_now()}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO experiments(id,path,body) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body",
                       (identifier, str(directory), json.dumps(record, ensure_ascii=False, allow_nan=False)))
            if export:
                self._csv(db)
        return record

    def records(self, category=None, kind=None):
        with self.connect() as db:
            records = [json.loads(row[0]) for row in db.execute("SELECT body FROM experiments")]
        return sorted([r for r in records if (not category or r.get("category") == category) and (not kind or r["kind"] == kind)],
                      key=lambda r: r.get("started_at") or r.get("updated_at", ""), reverse=True)

    def _csv(self, db):
        rows = []
        for (body,) in db.execute("SELECT body FROM experiments ORDER BY path"):
            r = json.loads(body)
            m = r["metrics"]
            percent = lambda name: round(m[name] * 100, 4) if isinstance(m.get(name), (float, int)) else ""
            rows.append({"id": r["id"], "kind": r["kind"], "category": r.get("category", ""), "variant": r.get("variant", ""),
                         "protocol": r.get("protocol", ""), "status": r.get("status", ""), "development": r.get("development", False),
                         "started_at": r.get("started_at", ""), "completed_at": r.get("completed_at", ""),
                         "accuracy_percent": percent("accuracy"), "macro_precision_percent": percent("macro_precision"),
                         "macro_recall_percent": percent("macro_recall"), "macro_f1_percent": percent("macro_f1"),
                         "samples": m.get("samples", ""), "successful_attempts_percent": percent("successful_attempt_rate"),
                         "mean_model_ms": m.get("latency", {}).get("mean_model_ms", ""),
                         "dataset_id": r.get("dataset_id", ""), "path": r["path"], "error": r.get("error", "")})
        write_csv(self.directory / "experiments.csv", rows, fields=list(rows[0]) if rows else ["id", "kind", "status", "path"])

    def reconcile(self, runs: Path):
        """Import pre-existing run/session files; the on-disk artifacts remain authoritative."""
        candidates = list(runs.rglob("run.json")) if runs.exists() else []
        testing = self.directory / "testing"
        candidates += list(testing.rglob("session.json")) if testing.exists() else []
        for file in candidates:
            try:
                state = read_json(file)
                path = file.parent
                if file.name == "session.json":
                    metrics = read_json(path / "summary.json") if (path / "summary.json").exists() else {}
                    kind, params = state.get("kind", "live_test"), state.get("parameters", {})
                elif state.get("protocol") == "loso" and (path / "loso-summary.json").exists():
                    summary = read_json(path / "loso-summary.json")
                    metrics = {**summary["pooled"], **{key: value["mean"] for key, value in summary["statistics"].items()}}
                    kind, params = "training", {}
                else:
                    metrics = read_json(path / "metrics.json") if (path / "metrics.json").exists() else {}
                    params = read_json(path / "training_config.json") if (path / "training_config.json").exists() else {}
                    kind = "training_fold" if "test_signer" in state else "training"
                self.save(path, kind, state, metrics, params, export=False)
            except (OSError, ValueError, KeyError):
                continue
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._csv(db)
        return self.records()
