import io
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

import pytest

from hudyat_train import cli, simulation
from hudyat_train.jobs import Job
from hudyat_train.storage import read_json, sha256, write_json


def exported_model(project, directory, category="alphabet"):
    directory.mkdir(parents=True)
    (directory / "weights.bin").write_bytes(b"synthetic export fixture")
    prep = {"features": 128, "normalization": "hand-relative-sequence-v1",
            "extraction": {"model_sha256": sha256(project.path("hand_model")), "detection_confidence": 0.5}}
    write_json(directory / "model.json", {"format": "fsl-conv1d-bilstm-v1", "inputShape": [32, 128],
                                         "category": category, "variant": "sequence", "version": "abc123",
                                         "classes": ["A", "B"], "weightSha256": sha256(directory / "weights.bin"),
                                         "preprocessing": prep, "development": True})
    return directory


@pytest.fixture
def browser_server(project, monkeypatch):
    model = exported_model(project, project.root / "export")
    published = simulation.prepare_release(project, model, project.root / "private", Job())
    web = project.root / "web"
    web.mkdir()
    (web / "index.html").write_text("browser simulator")
    hudyat = project.root / "HUDYAT"
    for name in ("mediapipe/selfie_segmenter.tflite", "sounds/correct.mp3", "sounds/wrong.mp3"):
        path = hudyat / "client/public" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"public fixture")
    reports = simulation.BrowserReports(project, published, {"files": {"actual-source.ts": "hash"}})
    server, stopped = simulation.create_server(web, published, None, reports, {"profile": "baseline"}, hudyat)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield url, reports, published, stopped
    finally:
        server.shutdown()
        server.server_close()
        reports.close()
        worker.join(timeout=2)


def fetch(url, body=None, headers=None):
    request = Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers or {})
    with urlopen(request, timeout=5) as response:
        return response.read()


def test_browser_server_loads_only_selected_model_and_verified_detector(browser_server):
    url, reports, published, _ = browser_server
    release = json.loads(fetch(url + "/api/models/alphabet"))["data"]
    assert release == published.release
    assert fetch(url + release["weightsUrl"]) == b"synthetic export fixture"
    assert fetch(url + release["handUrl"]) == b"hand detector fixture"
    assert fetch(url + "/") == b"browser simulator"
    for route in ("/api/models/family", "/../project.json", "/%2e%2e/project.json", "/api/models/files/models/alphabet/candidate.json"):
        with pytest.raises(HTTPError) as error:
            fetch(url + route)
        assert error.value.code == 404
    with pytest.raises(HTTPError) as error:
        fetch(url, headers={"Host": "other.example"})
    assert error.value.code == 403


def test_browser_results_require_same_origin_and_session_token(browser_server):
    url, reports, _, _ = browser_server
    config = json.loads(fetch(url + "/simulation/config"))
    body = {"configuration": str(uuid.uuid4()), "settings": {"profile": "baseline"},
            "summary": {"trials": 1, "recognized": 1, "wrongGuesses": 0}, "trials": [], "decisions": []}
    for headers in ({}, {"Origin": url}, {"Origin": "https://other.example", "X-Simulation-Token": config["token"]}):
        with pytest.raises(HTTPError) as error:
            fetch(url + "/simulation/snapshot", body, headers)
        assert error.value.code == 403
    headers = {"Origin": url, "X-Simulation-Token": config["token"]}
    result = json.loads(fetch(url + "/simulation/snapshot", body, headers))
    assert read_json(Path(result["saved"]) / "browser-results.json")["settings"] == {"profile": "baseline"}
    assert json.loads(fetch(url + "/simulation/stop", {}, headers))["stopped"]


def test_each_browser_configuration_retains_its_own_settings_and_scope(browser_server):
    url, reports, _, _ = browser_server
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    for identifier, profile, trials in ((first, "baseline", 2), (second, "enhanced", 3)):
        reports.save({"configuration": identifier, "settings": {"profile": profile},
                      "summary": {"trials": trials, "recognized": 1, "fps": 12},
                      "trials": [], "decisions": [], "camera": {"width": 1280, "height": 720},
                      "pixels": "must not be saved", "landmarks": ["must not be saved"]}, complete=True)
    for identifier, profile, trials in ((first, "baseline", 2), (second, "enhanced", 3)):
        directory = reports.directory / identifier
        snapshot = read_json(directory / "browser-results.json")
        state = read_json(directory / "session.json")
        assert snapshot["summary"]["trials"] == trials
        assert snapshot["settings"]["profile"] == profile
        assert "pixels" not in snapshot and "landmarks" not in snapshot
        assert "accuracy" not in read_json(directory / "summary.json")
        assert state["protocol"] == "hudyat_browser_simulation"
        assert "HUDYAT browser" in (directory / "report.html").read_text(encoding="utf-8")
        assert state["runtime_source"]["files"] == {"actual-source.ts": "hash"}


def test_browser_reports_reject_path_configuration_and_malformed_measurements(browser_server):
    _, reports, _, _ = browser_server
    for payload in ({"configuration": "../../outside"}, {"configuration": str(uuid.uuid4()), "settings": [], "summary": {}}):
        with pytest.raises(ValueError):
            reports.save(payload)


def test_candidate_reports_identify_the_selected_release(project, tmp_path):
    model = exported_model(project, project.root / "published")
    candidate_path = exported_model(project, project.root / "candidate")
    metadata = read_json(candidate_path / "model.json")
    metadata.update(version="candidate456", variant="static")
    write_json(candidate_path / "model.json", metadata)
    published = simulation.prepare_release(project, model, project.root / "staging-published", Job())
    candidate = simulation.prepare_release(project, candidate_path, project.root / "staging-candidate", Job())
    reports = simulation.BrowserReports(project, published, {"files": {}}, candidate)
    try:
        directory = reports.save({"configuration": str(uuid.uuid4()), "settings": {"profile": "custom", "alphabetModel": "candidate"},
                                  "summary": {"trials": 1}, "trials": [], "decisions": []})
        state = read_json(directory / "session.json")
        assert state["release"]["version"] == "candidate456"
        assert state["source_model"] == str(candidate_path)
        assert state["variant"] == "static"
    finally:
        reports.close()


def test_cli_passes_browser_simulation_options(project, monkeypatch):
    received = []
    monkeypatch.setattr(cli.sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))
    monkeypatch.setattr(simulation, "run_simulator", lambda *args: received.append(args))
    assert cli.main(["--root", str(project.root), "simulate", "--run", "selected-model", "--hudyat", str(project.root / "HUDYAT"),
                     "--candidate", "candidate-model", "--profile", "enhanced", "--settings-file", "settings.json",
                     "--target", "A", "--port", "8765", "--no-browser"]) == 0
    args = received[0]
    assert args[1] == project.root / "selected-model" and args[3] == project.root / "candidate-model"
    assert args[4:9] == ("enhanced", project.root / "settings.json", "A", 8765, False)
