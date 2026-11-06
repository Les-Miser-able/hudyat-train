import base64
from http.cookiejar import CookieJar
import io
import json
import re
import runpy
import threading
import time
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import pytest

from hudyat_train.experiments import Registry
from hudyat_train.storage import read_json, write_json
from hudyat_train.web_app import Studio, StudioServer
from tests.conftest import add_source


@pytest.fixture
def studio(project):
    return Studio(project.root)


@pytest.fixture
def http_studio(studio):
    server = StudioServer(studio)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = build_opener(HTTPCookieProcessor(CookieJar()))
    with client.open(server.origin + "/", timeout=5) as response:
        html = response.read().decode()
    token = re.search(r'name="studio-token" content="([^"]+)"', html)[1]
    def request(path, body=None, headers=None):
        options = {"Origin": server.origin, "X-Studio-Token": token, **(headers or {})}
        if body is not None:
            options["Content-Type"] = "application/json"
        req = Request(server.origin + path, data=json.dumps(body).encode() if body is not None else None, headers=options)
        with client.open(req, timeout=8) as response:
            return response.read(), response
    yield studio, server, request
    studio.stop()
    if studio.process:
        studio.process.wait(timeout=10)
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def wait_job(studio):
    deadline = time.monotonic() + 15
    while studio.process is not None and time.monotonic() < deadline:
        time.sleep(.02)
    assert studio.process is None
    return studio.status()


def test_http_studio_runs_real_scan_and_reads_counts(http_studio, project):
    studio, server, request = http_studio
    source = add_source(project, content=b"a unique raw recording")
    payload, response = request("/api/jobs", {"action": "scan", "version": "sequence"})
    assert response.status == 202
    assert json.loads(payload)["kind"] == "scan"
    status = wait_job(studio)
    assert status["job"]["status"] == "complete"
    state, _ = request("/api/state?version=sequence")
    state = json.loads(state)
    assert state["scan"]["sources"] == 1
    assert sum(c["signs"] for c in state["categories"]) == 108
    assert state["history"] == {}
    assert read_json(project.root / "reports/scan-sequence.json")["sources"] == 1
    assert source.read_bytes() == b"a unique raw recording"
    assert "png" not in (project.root / status["job"]["log_path"]).read_text(encoding="utf-8")


def test_local_http_requires_session_origin_and_token(http_studio):
    studio, server, request = http_studio
    with pytest.raises(HTTPError) as error:
        urlopen(server.origin + "/api/state", timeout=4)
    assert error.value.code == 403

    for headers in ({"Origin": "https://unrelated.example"}, {"X-Studio-Token": "invalid"}, {"Host": "rebinding.example"}):
        with pytest.raises(HTTPError) as error:
            request("/api/stop", {}, headers)
        assert error.value.code == 403
    assert studio.process is None


def test_study_validation_rejects_bad_inputs_before_starting_worker(http_studio):
    studio, server, request = http_studio
    for body in (
        {"action": "train", "epochs": 3},
        {"action": "train", "epochs": True},
        {"action": "train", "augmentation": -1},
        {"action": "train", "mirroring": "unknown"},
        {"action": "train", "development": "false"},
        {"action": "train", "category": "all"},
        {"action": "scan", "category": "alphabet; arbitrary command"},
        {"action": "arbitrary-command"},
        {"action": "export", "model": ""},
    ):
        with pytest.raises(HTTPError) as error:
            request("/api/jobs", body)
        assert error.value.code == 400
        assert studio.process is None


def test_results_history_artifacts_and_path_boundary(http_studio, project):
    studio, server, request = http_studio
    run = project.path("runs_dir") / "model with spaces"
    write_json(run / "history.json", {"accuracy": [.6, .8], "val_accuracy": [.5, .7]})
    write_json(run / "classes.json", ["A", "B"])
    (run / "attempts.csv").write_text("expected,predicted,confidence\nA,A,0.9\n", encoding="utf-8")
    record = Registry(project.root).save(run, "training", {"status": "complete", "category": "alphabet", "variant": "sequence", "development": True}, {"accuracy": .625})
    body, _ = request("/api/state?version=sequence")
    assert json.loads(body)["history"]["val_accuracy"] == [.5, .7]
    body, _ = request("/api/state?version=static")
    assert json.loads(body)["history"] == {}
    body, _ = request("/api/results?id=" + record["id"])
    assert json.loads(body)["attempts"][0]["expected"] == "A"
    body, response = request("/artifacts/" + record["id"] + "/attempts.csv")
    assert b"A,A,0.9" in body
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    with pytest.raises(HTTPError) as error:
        request("/artifacts/" + record["id"] + "/..%2f..%2fproject.json")
    assert error.value.code == 403
    assert studio.model_classes(str(run)) == ["A", "B"]
    write_json(run / "classes.json", {"invalid": "class list"})
    assert studio.model_classes(str(run)) == []


def test_live_worker_preview_is_memory_only_and_commands_are_forwarded(studio, project, monkeypatch):
    import hudyat_train.web_app as web_app
    model = project.root / "external model"
    write_json(model / "classes.json", ["A"])
    png = base64.b64encode(b"camera pixels in memory").decode()
    events = [{"type": "ready", "classes": ["A"]}, {"type": "frame", "png": png, "preview_fps": 30},
              {"type": "estimate", "predicted": "A", "confidence": .9}]
    done = threading.Event()
    class Process:
        stdout = io.StringIO("\n".join(json.dumps(event) for event in events) + "\n")
        stderr = io.StringIO("")
        stdin = io.StringIO()
        def wait(self, timeout=None):
            assert done.wait(timeout or 8)
            return 0
    process = Process()
    monkeypatch.setattr(web_app.subprocess, "Popen", lambda *args, **kwargs: process)
    studio.start("live", {"model": str(model), "mode": "manual"})
    deadline = time.monotonic() + 4
    while not studio.frame and time.monotonic() < deadline:
        time.sleep(.01)
    assert studio.frame == b"camera pixels in memory"
    assert "png" not in json.dumps(studio.status())
    with pytest.raises(ValueError, match="already running"):
        studio.start("scan", {})
    studio.control({"type": "target", "label": "A"})
    studio.control({"type": "capture"})
    studio.control({"type": "finish"})
    assert '"label": "A"' in process.stdin.getvalue()
    assert '"type": "capture"' in process.stdin.getvalue()
    with pytest.raises(ValueError):
        studio.control({"type": "target", "label": "not in classes"})
    studio.stop()
    assert studio.cancel_file.read_bytes() == b"stop\n"
    assert '"type": "stop"' in process.stdin.getvalue()
    done.set()
    status = wait_job(studio)
    saved = (project.root / status["job"]["log_path"]).read_text(encoding="utf-8")
    assert png not in saved and '"png"' not in saved
    assert studio.frame is None


def test_training_simulation_export_and_settings_reach_existing_cli(studio, project):
    model = project.root / "model"
    model.mkdir()
    executable, arguments, kind = studio._arguments("train", {"development": True, "epochs": 5, "augmentation": 3,
                                                              "mirroring": "approved", "protocol": "loso", "seed": 24})
    assert arguments[:1] == ["train"] and kind == "training"
    for key, expected in (("--epochs", "5"), ("--augmentation-copies", "3"), ("--mirroring", "approved"), ("--protocol", "loso"), ("--seed", "24")):
        assert arguments[arguments.index(key)+1] == expected
    assert "--development" in arguments
    _, arguments, kind = studio._arguments("simulate", {"model": str(model), "candidate": str(model), "profile": "enhanced", "simulation_settings": "lab.json", "target": "A"})
    assert kind == "simulation" and "--no-browser" in arguments and "--candidate" in arguments and "--settings-file" in arguments
    _, arguments, _ = studio._arguments("export", {"model": str(model), "allow_development": True, "output": "exports/static"})
    assert "--allow-development" in arguments and str(project.root / "exports/static") in arguments
    studio.save_tools({"training_python": executable, "hudyat_path": str(project.root / "HUDYAT")})
    assert read_json(project.root / "settings.local.json")["training_python"] == executable
    assert studio.tools()["hudyat_path"] == str(project.root / "HUDYAT")


def test_normal_launch_opens_browser_and_shuts_down_safely(project, monkeypatch):
    import hudyat_train.web_app as web_app
    opened = []
    monkeypatch.setattr(web_app.webbrowser, "open", lambda url: opened.append(url))
    thread = threading.Thread(target=web_app.main, args=(project.root,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 6
    while not opened and time.monotonic() < deadline:
        time.sleep(.01)
    assert opened
    url = opened[0]
    assert url.startswith("http://127.0.0.1:")
    assert (project.root / "tmp/studio-url.txt").read_text().strip() == url
    client = build_opener(HTTPCookieProcessor(CookieJar()))
    with client.open(url, timeout=4) as response:
        html = response.read().decode()
    token = re.search(r'name="studio-token" content="([^"]+)"', html)[1]
    request = Request(url + "api/shutdown", data=b"{}", headers={"Origin": url.rstrip("/"),
                      "X-Studio-Token": token, "Content-Type": "application/json"})
    with client.open(request, timeout=4) as response:
        assert response.status == 202
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_broken_saved_scan_does_not_block_the_studio(studio, project):
    path = project.root / "reports/scan-sequence.json"
    path.write_text("partial JSON", encoding="utf-8")
    snapshot = studio.snapshot()
    assert snapshot["scan"] is None
    assert snapshot["warnings"]
    assert len(snapshot["categories"]) == 9


def test_gui_live_and_launcher_entry_points_use_browser_studio(project, monkeypatch):
    from hudyat_train import cli, web_app
    from hudyat_train.config import ROOT
    calls = []
    monkeypatch.setattr(web_app, "main", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert cli.main(["--root", str(project.root), "gui"]) == 0
    assert calls[-1][0] == (project.root,)
    assert cli.main(["--root", str(project.root), "live", "--run", "models/chosen", "--camera", "2", "--mode", "manual"]) == 0
    assert calls[-1][1] == {"testing_run": project.root / "models/chosen", "camera": 2, "mode": "manual"}
    runpy.run_path(str(ROOT / "desktop.py"), run_name="__main__")
    assert calls[-1] == ((), {})
