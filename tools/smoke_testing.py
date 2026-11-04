"""Exercise result recording, Keras/export inference, and manual live-worker IPC.

Uses existing synthetic smoke runs and an explicitly selected video, never a webcam.
All scores here are development diagnostics, not study recognition results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hudyat_train.config import ROOT, load_settings
from hudyat_train.experiments import Registry
from hudyat_train.inference import Predictor, describe_model
from hudyat_train.storage import read_json, within, write_json
from hudyat_train.testing import automated_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT / "tmp" / "synthetic-final")
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.project)
    source = read_json(settings.root / "smoke-results.json")["results"]
    proof = {"synthetic_models_only": True, "physical_webcam_opened": False, "loaders": []}
    for result in source:
        bundle = Path(result["bundle"])
        catalog = read_json(bundle / "catalog.json")
        release = next(e for e in catalog["categories"] if e["id"] == result["category"])
        export = within(bundle, release["modelUrl"]).parent
        probes = read_json(export / "parity.json")
        sample = np.asarray(probes["inputs"][0], np.float32)
        for folder in (Path(result["run"]), export):
            info = describe_model(folder)
            predictor = Predictor(info)
            actual = predictor.predict(sample)
            errors = [abs(row["probability"] - probes["expected"][0][info.classes.index(row["label"])]) for row in actual["top3"]]
            if max(errors) >= 2e-4:
                raise AssertionError(f"Inference parity failed: {folder}: {max(errors)}")
            proof["loaders"].append({"category": info.category, "variant": info.variant, "exported": info.exported, "max_error": max(errors)})
    alphabet = next(row for row in source if row["category"] == "alphabet" and row["variant"] == "sequence")
    proof["automated_random_test"] = str(automated_test(settings, Path(alphabet["run"])))
    loso = next((p for p in (settings.path("runs_dir") / "alphabet/sequence").glob("*-loso") if (p / "loso-summary.json").exists()), None)
    if loso is None:
        from hudyat_train.training import TrainOptions, train
        loso = train(settings, TrainOptions("alphabet", protocol="loso", epochs=1, development=True))
    proof["automated_loso_test"] = str(automated_test(settings, loso))
    command = [sys.executable, "-u", "-m", "hudyat_train", "--root", str(settings.root), "live-worker",
               "--run", alphabet["run"], "--video", str(args.video.resolve()), "--mode", "manual", "--target", "A",
               "--max-attempts", "1", "--control-stdin"]
    error_log = settings.root / "reports" / "manual-smoke.stderr.log"
    manual_rows, first_hand_at, sent_finish, preview_seen, session_path = [], None, False, False, None
    with error_log.open("w", encoding="utf-8") as errors:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                   text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "ready":
                    process.stdin.write('{"type":"capture"}\n')
                    process.stdin.flush()
                elif event.get("type") == "frame":
                    preview_seen |= bool(event.get("png"))
                    if event["left"] or event["right"]:
                        first_hand_at = event["at"] if first_hand_at is None else first_hand_at
                        if not sent_finish and event["at"] - first_hand_at >= 0.8:
                            process.stdin.write('{"type":"finish"}\n')
                            process.stdin.flush()
                            sent_finish = True
                elif event.get("type") == "attempt":
                    manual_rows.append(event["row"])
                elif event.get("type") == "session_complete":
                    session_path = event["path"]
            code = process.wait(timeout=60)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    if code or not preview_seen or not sent_finish or len(manual_rows) != 1 or manual_rows[0]["mode"] != "manual" or manual_rows[0].get("predicted") is None:
        raise AssertionError(f"Manual live-worker IPC failed: exit={code}, preview={preview_seen}, finish={sent_finish}, rows={manual_rows}; see {error_log}")
    proof["manual_video_replay"] = {"path": session_path, "preview_received": preview_seen,
                                      "manual_finish_command": sent_finish, "attempts": len(manual_rows)}
    registry = Registry(settings.root)
    proof["recorded_experiments"] = len(registry.reconcile(settings.path("runs_dir")))
    output = settings.root / "reports" / "testing-smoke-validation.json"
    write_json(output, proof)
    print("Testing workflow smoke passed:", output)


if __name__ == "__main__":
    main()
