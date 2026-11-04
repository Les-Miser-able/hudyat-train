"""Automated held-out tests and durable logs for manually labelled live attempts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import uuid

from .experiments import Registry, append_event, runtime_details, utc_now, write_csv
from .inference import ModelInfo, Predictor, describe_model
from .jobs import Job
from .storage import fingerprint, read_json, sha256, within, write_json

ATTEMPT_FIELDS = ["attempt", "at", "mode", "source", "signer", "fold", "expected", "predicted", "confidence", "margin",
                  "accepted", "outcome", "model_ms", "pipeline_ms", "error"]


def summarize_attempts(rows, classes, threshold=0.85):
    import numpy as np
    from .evaluation import classification_report
    valid = [r for r in rows if r.get("predicted") is not None]
    labelled = [r for r in rows if r.get("expected") in classes]
    scored = [r for r in labelled if r.get("predicted") in classes]
    if scored:
        indices = {label: i for i, label in enumerate(classes)}
        result = classification_report([indices[r["expected"]] for r in scored], [indices[r["predicted"]] for r in scored], classes)
        tested = [r for r in result["per_class"] if r["support"] > 0]
        for name in ("precision", "recall", "f1"):
            result["macro_" + name] = float(np.mean([r[name] for r in tested]))
        # A live session with partial coverage cannot claim the study benchmark.
        result.pop("benchmark", None)
    else:
        result = {"accuracy": None, "macro_precision": None, "macro_recall": None, "macro_f1": None,
                  "samples": 0, "per_class": [], "confusion_matrix": []}
    tested_labels = {r["expected"] for r in scored}
    timings = [r["model_ms"] for r in valid if isinstance(r.get("model_ms"), (int, float))]
    pipelines = [r["pipeline_ms"] for r in valid if isinstance(r.get("pipeline_ms"), (int, float))]
    result.update(attempts=len(rows), labelled_attempts=len(labelled), valid_predictions=len(valid),
                  invalid_attempts=len(rows) - len(valid), rejected_predictions=sum(not r["accepted"] for r in valid),
                  unlabelled_attempts=len(rows) - len(labelled), confidence_threshold=threshold,
                  successful_attempt_rate=sum(r["outcome"] == "correct" for r in labelled) / len(labelled) if labelled else None,
                  tested_classes=sorted(tested_labels), untested_classes=[c for c in classes if c not in tested_labels],
                  macro_definition="Mean over classes with ground-truth support; untested classes are listed separately.",
                  latency={"scope": "Python model call only, after warm-up; excludes camera, preprocessing, and GUI rendering",
                           "mean_model_ms": float(np.mean(timings)) if timings else None,
                           "median_model_ms": float(np.median(timings)) if timings else None,
                           "p95_model_ms": float(np.percentile(timings, 95)) if timings else None,
                           "mean_pipeline_ms": float(np.mean(pipelines)) if pipelines else None,
                           "pipeline_scope": "Python input preparation + model; excludes capture, detector, GUI rendering and browser"})
    return result


class TestSession:
    def __init__(self, settings, info: ModelInfo, kind: str, parameters=None):
        self.settings, self.info, self.rows = settings, info, []
        self.threshold = float(info.preprocessing.get("confidence_threshold", 0.85))
        if not 0 < self.threshold <= 1:
            raise ValueError("Invalid model confidence threshold.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.directory = settings.root / "reports" / "testing" / info.category / info.variant / f"{stamp}-{kind}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.state = {"status": "running", "kind": kind, "category": info.category, "variant": info.variant,
                      "development": info.development, "protocol": "held_out_replay" if kind == "automated_test" else "live_exploratory",
                      "source_model": str(info.directory), "checkpoint_sha256": info.checkpoint_sha256,
                      "started_at": utc_now(), "parameters": parameters or {}, "classes": info.classes,
                      "runtime": runtime_details(), "preprocessing": info.preprocessing,
                      "model_signature": fingerprint({"weights": info.checkpoint_sha256, "classes": info.classes,
                                                       "preprocessing": info.preprocessing})}
        self.registry = Registry(settings.root)
        append_event(self.directory / "events.jsonl", {"type": "started", **self.state})
        self._save()

    def _save(self):
        summary = summarize_attempts(self.rows, self.info.classes, self.threshold)
        write_json(self.directory / "session.json", self.state)
        write_json(self.directory / "attempts.json", self.rows)
        write_csv(self.directory / "attempts.csv", self.rows, ATTEMPT_FIELDS)
        write_json(self.directory / "summary.json", summary)
        self.registry.save(self.directory, self.state["kind"], self.state, summary, self.state["parameters"])
        return summary

    def add(self, result=None, expected=None, mode="automatic", source="webcam", error=None, **extra):
        if expected is not None and expected not in self.info.classes:
            raise ValueError("Expected sign is not one of this model's classes.")
        result = result or {}
        predicted, confidence = result.get("predicted"), result.get("confidence")
        accepted = predicted is not None and confidence is not None and confidence >= self.threshold
        outcome = "invalid" if predicted is None else "rejected" if not accepted else "unscored" if expected is None else "correct" if predicted == expected else "incorrect"
        row = {"attempt": len(self.rows) + 1, "at": utc_now(), "mode": mode, "source": source,
               "expected": expected, "predicted": predicted, "confidence": confidence, "accepted": accepted,
               "outcome": outcome, "model_ms": result.get("model_ms"), "margin": result.get("margin"),
               "pipeline_ms": result.get("pipeline_ms"), "error": error, **extra}
        if result.get("top3"):
            row["top3"] = result["top3"]
        self.rows.append(row)
        append_event(self.directory / "events.jsonl", {"type": "attempt", **row})
        self._save()
        return row

    def close(self, status="complete", error=None):
        self.state.update(status="saving_reports", completed_at=utc_now())
        if error:
            self.state["error"] = error
        summary = self._save()
        from .reporting import result_report
        try:
            result_report(self.directory, {**self.state, "status": status}, summary, self.rows)
        except BaseException as failure:
            self.state.update(status="failed", error=f"Report generation failed: {failure}")
            self._save()
            raise
        self.state["status"] = status
        self._save()
        append_event(self.directory / "events.jsonl", {"type": "finished", "status": status, "error": error})
        return summary


def _sample(settings, info, manifest, item):
    import numpy as np
    candidates = [settings.root] + [p for p in info.directory.parents if (p / "project.json").is_file()]
    for root in dict.fromkeys(candidates):
        base = within(root, manifest["settings"]["extracted_dir"]) / info.variant / info.category
        path = within(base, item["sample"])
        if path.is_file() and sha256(path) == item["sample_sha256"]:
            sample = np.load(path, allow_pickle=False)
            if sample.dtype != np.float32 or sample.shape != (32, info.features) or not np.isfinite(sample).all():
                raise ValueError(f"Invalid test sample: {item['sample']}")
            return sample
    raise ValueError(f"Recorded test sample is missing or changed: {item['sample']}. Restore the original extracted dataset.")


def automated_test(settings, run: Path, job: Job | None = None):
    job = job or Job()
    run = run.resolve()
    state = read_json(run / "run.json")
    if state.get("status") != "complete":
        raise ValueError("Automated testing requires a completed training run.")
    folders = [run]
    if state.get("protocol") == "loso" and not (run / "best.keras").exists():
        expected = read_json(run / "loso-summary.json")["folds"]
        folders = [run / ("test-" + fold["test_signer"]) for fold in expected]
    infos = [describe_model(folder) for folder in folders]
    if any(info.classes != infos[0].classes for info in infos):
        raise ValueError("LOSO fold class orders disagree.")
    work = []
    held_out_digests = set()
    for info in infos:
        splits = read_json(info.directory / "splits.json")
        manifest = read_json(info.directory / "dataset-index.json")
        test = splits.get("test", [])
        if not test:
            raise ValueError("The recorded test split is empty.")
        forbidden = {it["digest"] for part in ("train", "validation") for it in splits[part]}
        if any(it["digest"] in forbidden for it in test):
            raise ValueError("Held-out test sources overlap with training/validation.")
        if any(it["label"] not in info.classes for it in test):
            raise ValueError("A held-out label is not in the evaluated class ordering.")
        if held_out_digests & {it["digest"] for it in test}:
            raise ValueError("A source appears in multiple held-out folds.")
        held_out_digests.update(it["digest"] for it in test)
        work.append((info, [(item, _sample(settings, info, manifest, item)) for item in test]))
    session = TestSession(settings, infos[0], "automated_test", {"source_run": str(run), "folds": len(infos), "split": "test", "warm_up_excluded": True})
    session.state["dataset_id"] = fingerprint([[item["digest"] for item, _ in samples] for _, samples in work])[:16]
    session.state["source_models"] = [{"path": str(info.directory), "checkpoint_sha256": info.checkpoint_sha256,
                                       "test_signer": info.state.get("test_signer")} for info in infos]
    try:
        for info, samples in work:
            job.check()
            predictor = Predictor(info)
            for i, (item, sample) in enumerate(samples, 1):
                job.check()
                started = perf_counter()
                result = predictor.predict(sample)
                result["pipeline_ms"] = (perf_counter() - started) * 1000
                row = session.add(result, expected=item["label"], mode="held_out", source=item["relative"],
                                  signer=item["signer"], fold=info.state.get("test_signer", ""))
                job.emit(f"[{i}/{len(samples)}] {item['label']} -> {row['predicted']} ({row['confidence']:.1%}, {row['outcome']}, {row['model_ms']:.1f} ms)")
        summary = session.close()
        job.emit(f"Automated test complete: {summary['samples']} labelled predictions. Report: {session.directory / 'report.html'}")
        job.emit(f"TEST_PATH={session.directory}")
        return session.directory
    except BaseException as error:
        session.close("interrupted" if error.__class__.__name__ in ("Cancelled", "KeyboardInterrupt") else "failed", str(error))
        raise
