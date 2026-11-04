"""Serve a local test shell around HUDYAT's actual browser recognition modules."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import escape
import json
import mimetypes
from pathlib import Path
import re
import shutil
import subprocess
import threading
from urllib.parse import unquote, urlsplit
import uuid
import webbrowser

from .catalog import CATEGORY_NAMES
from .config import ROOT
from .experiments import Registry, append_event, write_csv
from .inference import describe_model, detector_paths
from .jobs import Job
from .storage import read_json, sha256, within, write_json


SIMULATOR = ROOT / "simulator"
SCOPE = "HUDYAT browser camera and recognition code; exploratory trials on this browser/device, separate from held-out classification accuracy."


@dataclass
class BrowserRelease:
    info: object
    release: dict
    assets: dict[str, Path]
    source_directory: Path


def prepare_release(settings, model: Path, output: Path, job: Job) -> BrowserRelease:
    info = describe_model(model)
    source_directory = info.directory
    if not info.exported:
        from .exporting import export_run
        job.emit("Preparing a private browser bundle for this simulation...")
        bundle = export_run(settings, info.directory, output, allow_development=True, emit=job.emit)
        entry = next(row for row in read_json(bundle / "catalog.json")["categories"] if row["id"] == info.category)
        info = describe_model(within(bundle, entry["modelUrl"]).parent)
    paths = detector_paths(settings, info)
    metadata = read_json(info.directory / "model.json")
    prefix = f"/api/models/files/simulation/{metadata['version']}"
    release = {"id": info.category, "name": CATEGORY_NAMES[info.category], "version": metadata["version"],
               "modelUrl": prefix + "/model.json", "weightsUrl": prefix + "/weights.bin",
               "modelSha256": sha256(info.directory / "model.json"), "weightsSha256": info.checkpoint_sha256}
    assets = {release["modelUrl"]: info.directory / "model.json", release["weightsUrl"]: info.directory / "weights.bin"}
    for kind, path in paths.items():
        release[kind + "Url"] = prefix + "/" + kind + ".task"
        release[kind + "Sha256"] = sha256(path)
        assets[release[kind + "Url"]] = path
    return BrowserRelease(info, release, assets, source_directory)


def build_simulator(hudyat: Path, output: Path, job: Job):
    node = shutil.which("node")
    if not node or not (SIMULATOR / "node_modules/vite/package.json").is_file():
        raise ValueError("The browser simulator needs Node.js 24 and its packages. Run setup-simulator.cmd in the trainer folder first.")
    job.check()
    job.emit("Building HUDYAT's current camera, Recognition lab, and inference code...")
    command = [node, str(SIMULATOR / "build.mjs"), "--hudyat", str(hudyat.resolve()), "--output", str(output)]
    process = subprocess.Popen(command, cwd=SIMULATOR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for line in process.stdout:
            job.emit(line.rstrip())
            job.check()
        if process.wait():
            raise ValueError("Could not build the HUDYAT simulator. See the build log above.")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
    job.check()
    return read_json(output / "hudyat-source.json")


class BrowserReports:
    """Save each settings configuration separately; no camera pixels or landmarks."""
    def __init__(self, settings, published: BrowserRelease, source: dict, candidate: BrowserRelease | None = None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.directory = settings.root / "reports/testing" / published.info.category / published.info.variant / (stamp + "-hudyat-browser")
        self.directory.mkdir(parents=True)
        self.registry, self.lock = Registry(settings.root), threading.RLock()
        self.configurations, self.closed = {}, False
        self.models = {name: {"source_model": str(model.source_directory), "browser_model": str(model.info.directory),
                              "checkpoint_sha256": model.info.checkpoint_sha256, "variant": model.info.variant,
                              "release": model.release} for name, model in (("published", published), ("candidate", candidate)) if model}
        self.base = {"category": published.info.category, "variant": published.info.variant,
                     "development": published.info.development, "kind": "hudyat_simulation",
                     "protocol": "hudyat_browser_simulation", "started_at": datetime.now(timezone.utc).isoformat(),
                     "source_model": str(published.source_directory), "checkpoint_sha256": published.info.checkpoint_sha256,
                     "source_models": self.models,
                     "measurement_scope": SCOPE, "runtime_source": source}
        write_json(self.directory / "hudyat-source.json", source)
        self._root("running")

    def _root(self, status):
        state = {**self.base, "status": status, "configurations": list(self.configurations)}
        write_json(self.directory / "session.json", state)
        write_json(self.directory / "summary.json", {"configurations": len(self.configurations)})
        rows = [{"configuration": key, "profile": snapshot["settings"].get("profile"),
                 "condition": snapshot.get("condition", ""), "trials": snapshot["summary"].get("trials", 0),
                 "recognized": snapshot["summary"].get("recognized", 0)} for key, snapshot in self.configurations.items()]
        write_csv(self.directory / "configurations.csv", rows, ["configuration", "profile", "condition", "trials", "recognized"])
        links = "".join(f"<li><a href='{row['configuration']}/report.html'>{escape(str(row['profile']))}: {escape(str(row['condition'])) or row['configuration']}</a></li>" for row in rows)
        (self.directory / "report.html").write_text("<!doctype html><meta charset='utf-8'><title>HUDYAT browser simulation</title>"
                                                    "<h1>HUDYAT browser simulation</h1><p>" + escape(SCOPE) + "</p><ul>" + links + "</ul>", encoding="utf-8")
        self.registry.save(self.directory, "hudyat_simulation", state)

    def save(self, payload, complete=False):
        with self.lock:
            if self.closed:
                raise ValueError("This simulation has ended.")
            identifier = payload.get("configuration", "")
            if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9-]{36}", identifier):
                raise ValueError("Invalid simulation configuration ID.")
            if not isinstance(payload.get("settings"), dict) or not isinstance(payload.get("summary"), dict):
                raise ValueError("Settings and browser measurements are required.")
            if not isinstance(payload.get("trials", []), list) or not isinstance(payload.get("decisions", []), list):
                raise ValueError("Invalid browser results.")
            # Deliberately omit any unexpected fields, including camera images and full landmark arrays.
            snapshot = {key: payload[key] for key in ("configuration", "settings", "summary", "trials", "decisions", "device", "camera", "condition") if key in payload}
            directory = self.directory / identifier
            write_json(directory / "browser-results.json", snapshot)
            summary = snapshot["summary"]
            metrics = {key: summary.get(key, 0) for key in ("trials", "recognized", "wrongGuesses", "fps", "captureMs", "segmentMs", "predictMs", "msToCorrect")}
            state = {**self.base, "status": "complete" if complete else "running", "parameters": {"condition": snapshot.get("condition", ""), **snapshot["settings"]},
                     "runtime": {"backend": "HUDYAT TensorFlow.js + MediaPipe Tasks", "device": snapshot.get("device"),
                                 "camera": snapshot.get("camera"), "delegate": summary.get("delegate")}}
            model_key = "candidate" if self.base["category"] == "alphabet" and snapshot["settings"].get("alphabetModel") == "candidate" else "published"
            state.update(self.models.get(model_key, {"source_model": None, "checkpoint_sha256": None, "model_status": "unavailable candidate"}))
            write_json(directory / "session.json", state)
            write_json(directory / "summary.json", metrics)
            write_csv(directory / "trials.csv", snapshot.get("trials", []), ["target", "msToCorrect", "wrongGuesses"])
            write_csv(directory / "decisions.csv", snapshot.get("decisions", []),
                      ["at", "target", "prediction", "score", "margin", "accepted", "reason", "signMs", "longestGapMs"])
            self.configurations[identifier] = snapshot
            self.registry.save(directory, "hudyat_simulation", state, metrics, state["parameters"])
            if complete:
                from .reporting import result_report
                result_report(directory, state, metrics)
            self._root("running")
            return directory

    def event(self, payload):
        with self.lock:
            if not self.closed:
                event = {key: payload[key] for key in ("configuration", "type", "target", "prediction", "score", "margin", "accepted", "reason", "at", "signMs", "longestGapMs") if key in payload}
                append_event(self.directory / "events.jsonl", event)

    def close(self, status="complete"):
        with self.lock:
            for snapshot in list(self.configurations.values()):
                self.save(snapshot, complete=True)
            self.closed = True
            self._root(status)


def create_server(web: Path, published: BrowserRelease, candidate: BrowserRelease | None, reports: BrowserReports,
                  config: dict, hudyat: Path, port=0):
    assets = dict(published.assets)
    if candidate:
        assets.update(candidate.assets)
    public = hudyat / "client/public"
    wasm = SIMULATOR / "node_modules/@mediapipe/tasks-vision/wasm"
    for path in wasm.glob("*"):
        if path.is_file():
            assets["/mediapipe/wasm/" + path.name] = path
    for name in ("mediapipe/selfie_segmenter.tflite", "sounds/correct.mp3", "sounds/wrong.mp3"):
        path = public / name
        if not path.is_file():
            raise ValueError(f"HUDYAT simulator asset is missing: {path}")
        assets["/" + name] = path
    token, stopped = uuid.uuid4().hex, threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                pass

        def log_message(self, *args):
            pass

        def response(self, value, status=200):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # A browser tab may close while its final snapshot is being saved.

        def host_ok(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.host_ok():
                self.response({"error": "Use the simulator's localhost URL."}, 403)
                return
            route = unquote(urlsplit(self.path).path)
            if route == "/simulation/config":
                self.response({**config, "token": token, "report_directory": str(reports.directory),
                               "model": published.release, "classes": published.info.classes,
                               "candidate_available": candidate is not None})
                return
            if route == "/simulation/ping":
                self.response({"running": not stopped.is_set()})
                return
            if route == "/api/models/" + published.info.category:
                self.response({"data": published.release})
                return
            if route == "/api/models/files/models/alphabet/candidate.json":
                self.response(candidate.release if candidate else {"error": "Select a candidate model in the trainer first."}, 200 if candidate else 404)
                return
            path = assets.get(route)
            if path is None:
                try:
                    path = within(web, "index.html" if route == "/" else route.lstrip("/"))
                except ValueError:
                    self.response({"error": "Unknown asset."}, 404)
                    return
            if not path.is_file():
                self.response({"error": "Unknown asset."}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", {".js": "text/javascript", ".wasm": "application/wasm", ".tflite": "application/octet-stream"}.get(path.suffix, mimetypes.guess_type(path.name)[0] or "application/octet-stream"))
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, self.wfile)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            origin = f"http://{self.headers.get('Host')}"
            if not self.host_ok() or self.headers.get("Origin") != origin or self.headers.get("X-Simulation-Token") != token:
                self.response({"error": "Use this simulator page to save results."}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2_000_000:
                    raise ValueError("Invalid browser results size.")
                payload = json.loads(self.rfile.read(size), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Invalid number.")))
                if not isinstance(payload, dict):
                    raise ValueError("Expected browser results.")
                route = urlsplit(self.path).path
                if route in ("/simulation/snapshot", "/simulation/save"):
                    directory = reports.save(payload, complete=route.endswith("save"))
                    self.response({"saved": str(directory)})
                elif route == "/simulation/event":
                    reports.event(payload)
                    self.response({"saved": True})
                elif route == "/simulation/stop":
                    stopped.set()
                    self.response({"stopped": True})
                else:
                    self.response({"error": "Unknown action."}, 404)
            except (ValueError, TypeError, KeyError) as error:
                self.response({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 0.25
    return server, stopped


def run_simulator(settings, model: Path, hudyat: Path, candidate=None, profile="baseline", settings_file=None,
                  target=None, port=0, open_browser=True, job=None):
    job = job or Job()
    if profile not in ("baseline", "enhanced") or not 0 <= port <= 65535:
        raise ValueError("Use baseline/enhanced and a valid local port.")
    work = settings.root / "tmp/hudyat-simulator" / uuid.uuid4().hex
    source = build_simulator(hudyat, work / "web", job)
    published = prepare_release(settings, model, work / "published", job)
    contender = prepare_release(settings, candidate, work / "candidate", job) if candidate else None
    if contender and (published.info.category != "alphabet" or contender.info.category != "alphabet"):
        raise ValueError("HUDYAT's candidate switch is available only for Alphabet models.")
    overrides = read_json(settings_file) if settings_file else None
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Import a settings object or HUDYAT Recognition lab results JSON.")
    if target is not None and target not in published.info.classes:
        raise ValueError("Expected sign is not included in the selected model.")
    job.check()
    reports = BrowserReports(settings, published, source, contender)
    server = None
    status = "complete"
    try:
        server, stopped = create_server(work / "web", published, contender, reports,
                                         {"profile": profile, "settings": overrides, "target": target}, hudyat, port)
        url = f"http://127.0.0.1:{server.server_port}/?lab=1"
        job.emit("SIMULATION_URL=" + url)
        job.emit("Browser reports: " + str(reports.directory))
        job.emit("Using HUDYAT's actual browser pipeline. Start the camera in the browser; no camera pixels are saved.")
        if open_browser:
            webbrowser.open(url)
        while not stopped.is_set():
            job.check()
            server.handle_request()
    except BaseException as error:
        status = "interrupted" if error.__class__.__name__ in ("Cancelled", "KeyboardInterrupt") else "failed"
        raise
    finally:
        if server:
            server.server_close()
        reports.close(status)
    return reports.directory
