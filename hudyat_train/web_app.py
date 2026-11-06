"""Local browser studio, using the same CLI workers and saved artifacts as the trainer."""
from __future__ import annotations

import argparse
import base64
from collections import deque
import csv
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import uuid
import webbrowser

from .catalog import Catalog, CATEGORY_NAMES
from .cli import initialize
from .config import ROOT, load_settings
from .experiments import Registry
from .storage import read_json, write_bytes, write_json

ASSETS = Path(__file__).with_name("web_ui")
TOOL_KEYS = ("extraction_python", "training_python", "hudyat_path", "tfjs_path")


def choice(data, key, options, default):
    value = data.get(key, default)
    if value not in options:
        raise ValueError(f"Invalid {key}: choose {', '.join(map(str, options))}.")
    return value


def integer(data, key, default, minimum, maximum):
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be a whole number from {minimum} to {maximum}.")
    return value


def flag(data, key, default=False):
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false.")
    return value


def text_field(data, key, default=""):
    value = data.get(key, default)
    if not isinstance(value, str) or len(value) > 4096 or "\x00" in value:
        raise ValueError(f"Invalid {key}.")
    return value.strip()


class Studio:
    def __init__(self, root=ROOT, testing_run=None, camera=0, mode="automatic"):
        self.settings = load_settings(root)
        initialize(self.settings)
        self.root = self.settings.root
        self.catalog = Catalog(self.settings.path("signs_file"))
        self.registry = Registry(self.root)
        self.registry.reconcile(self.settings.path("runs_dir"))
        self.lock = threading.RLock()
        self.frames = threading.Condition(self.lock)
        self.process = None
        self.job = {"id": None, "status": "idle", "kind": "", "returncode": None}
        self.logs = deque(maxlen=2000)
        self.sequence = 0
        self.revision = 1
        self.frame, self.frame_version = None, 0
        self.live = {}
        self.simulator_url = ""
        self.cancel_file = None
        self.closing = False
        self.finished = threading.Event()
        self.finished.set()
        self.initial = {"model": str(testing_run or ""), "camera": camera, "mode": mode}

    def tools(self):
        path = self.root / "settings.local.json"
        saved = read_json(path) if path.exists() else {}
        return {key: saved.get(key, str(self.root.parent / "HUDYAT") if key == "hudyat_path" else "") for key in TOOL_KEYS}

    def path(self, value):
        path = Path(value).expanduser()
        return (self.root / path).resolve() if not path.is_absolute() else path.resolve()

    def _arguments(self, action, data):
        tools = self.tools()
        if "tools" in data:
            if not isinstance(data["tools"], dict):
                raise ValueError("Invalid local tool settings.")
            tools.update({key: text_field(data["tools"], key, tools[key]) for key in TOOL_KEYS})
        variant = choice(data, "version", ("sequence", "static"), "sequence")
        category = choice(data, "category", ("all", *CATEGORY_NAMES), "alphabet")
        arguments = [action]
        kind = action
        if action == "scan":
            arguments += ["--variant", variant]
        elif action == "extract":
            arguments += ["--category", category, "--variant", variant]
            kind = "extraction"
        elif action == "train":
            if category == "all":
                raise ValueError("Choose one category to train.")
            protocol = choice(data, "protocol", ("random", "loso"), "random")
            epochs = integer(data, "epochs", 150, 1, 150)
            copies = integer(data, "augmentation", 5, 0, 100)
            seed = integer(data, "seed", 42, 0, 2**32 - 1)
            mirroring = choice(data, "mirroring", ("all", "approved", "off"), "all")
            development = flag(data, "development")
            if not development and (epochs != 150 or copies != 5):
                raise ValueError("Enable Development run for fewer epochs or non-default augmentation.")
            arguments += ["--category", category, "--variant", variant, "--protocol", protocol, "--epochs", str(epochs),
                          "--seed", str(seed), "--augmentation-copies", str(copies), "--mirroring", mirroring]
            if development:
                arguments += ["--development"]
            kind = "training"
        elif action in ("test", "live", "simulate", "export"):
            model = text_field(data, "model")
            if not model or not self.path(model).is_dir():
                raise ValueError("Choose an existing training run or exported model directory.")
            arguments += ["--run", str(self.path(model))]
            if action == "test":
                kind = "testing"
            elif action == "live":
                arguments[0] = "live-worker"
                arguments += ["--camera", str(integer(data, "camera", 0, 0, 100)), "--mode",
                              choice(data, "mode", ("automatic", "manual"), "automatic"), "--control-stdin"]
            elif action == "simulate":
                arguments += ["--hudyat", tools["hudyat_path"], "--profile", choice(data, "profile", ("baseline", "enhanced"), "baseline"), "--no-browser"]
                for field, option in (("candidate", "--candidate"), ("simulation_settings", "--settings-file")):
                    value = text_field(data, field)
                    if value:
                        arguments += [option, str(self.path(value))]
                kind = "simulation"
            elif action == "export":
                output = text_field(data, "output") or str(self.settings.path("exports_dir") / variant)
                arguments += ["--output", str(self.path(output))]
                if flag(data, "allow_development"):
                    arguments += ["--allow-development"]
            target = text_field(data, "target")
            if target and action in ("live", "simulate"):
                arguments += ["--target", target]
        elif action == "verify":
            output = text_field(data, "output") or str(self.settings.path("exports_dir") / variant)
            arguments += ["--hudyat", tools["hudyat_path"], "--bundle", str(self.path(output))]
            if tools["tfjs_path"]:
                arguments += ["--tfjs", tools["tfjs_path"]]
        elif action == "audit":
            arguments += ["--hudyat", tools["hudyat_path"]]
        elif action == "doctor":
            kind = choice(data, "interpreter", ("training", "extraction"), "training")
        elif action == "results":
            if flag(data, "rebuild"):
                arguments += ["--rebuild"]
            kind = "reporting"
        else:
            raise ValueError("Unknown trainer action.")
        configured = tools["training_python"] if kind in ("training", "testing", "live", "simulation", "export", "reporting") else tools["extraction_python"]
        executable = Path(configured or sys.executable)
        if executable.name.lower() == "pythonw.exe" and executable.with_name("python.exe").exists():
            executable = executable.with_name("python.exe")
        return str(executable), arguments, kind

    def start(self, action, data):
        executable, arguments, kind = self._arguments(action, data)
        with self.lock:
            if self.closing:
                raise ValueError("The trainer is shutting down.")
            if self.process is not None:
                raise ValueError("A job is already running. Stop it before starting another.")
            identifier = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            cancel_file = self.root / "tmp" / "jobs" / (identifier + ".cancel")
            cancel_file.parent.mkdir(parents=True, exist_ok=True)
            command = [executable, "-u", "-m", "hudyat_train", "--root", str(self.root), "--cancel-file", str(cancel_file), *arguments]
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE if kind == "live" else subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE if kind == "live" else subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace",
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.process, self.cancel_file = process, cancel_file
            self.finished.clear()
            self.logs.clear()
            self.job = {"id": identifier, "kind": kind, "action": action, "status": "running", "arguments": arguments,
                        "returncode": None, "category": data.get("category", "alphabet"), "version": data.get("version", "sequence")}
            self.live = {}
            self.frame = None
            self.simulator_url = ""
            self.revision += 1
            log_path = self.root / "reports" / "jobs" / (identifier + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.job["log_path"] = str(log_path)
            self._line("> " + " ".join(arguments))
            threading.Thread(target=self._collect, args=(process, kind, log_path), daemon=True).start()
            return dict(self.job)

    def _line(self, line):
        with self.lock:
            self.sequence += 1
            self.logs.append({"sequence": self.sequence, "text": line.rstrip()})
            if line.startswith("Epoch "):
                self.revision += 1
                self.job["progress"] = line.rstrip()
            elif line.startswith("RUN_PATH="):
                self.job["run_path"] = line.strip().split("=", 1)[1]
            elif line.startswith("SIMULATION_URL="):
                url = line.strip().split("=", 1)[1]
                if url.startswith("http://127.0.0.1:"):
                    self.simulator_url = url

    def _live_event(self, event):
        kind = event.get("type")
        with self.frames:
            if kind == "frame":
                png = event.get("png")
                if png:
                    try:
                        self.frame = base64.b64decode(png, validate=True)
                        self.frame_version += 1
                        self.frames.notify_all()
                    except (ValueError, TypeError):
                        pass
                self.live["diagnostics"] = {key: value for key, value in event.items() if key not in ("png", "type")}
            elif kind == "estimate":
                self.live["estimate"] = event
            elif kind == "attempt":
                self.live["attempt"] = event["row"]
                self.live["attempt_count"] = self.live.get("attempt_count", 0) + 1
            elif kind == "ready":
                self.live["ready"] = True
                self.live["classes"] = event["classes"]
            elif kind == "status":
                self.live["message"] = event["message"]
            elif kind == "session_complete":
                self.live["saved_path"] = event["path"]

    def _collect(self, process, kind, log_path):
        log_lock = threading.Lock()
        def consume(stream, saved, live=False):
            for line in stream:
                if live:
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError("Invalid live event")
                        self._live_event(event)
                        if event.get("type") in ("frame", "estimate"):
                            continue
                    except (ValueError, TypeError, KeyError):
                        if '"png"' in line:
                            continue
                with log_lock:
                    saved.write(line)
                    saved.flush()
                self._line(line)
        try:
            with log_path.open("w", encoding="utf-8") as saved:
                stderr = None
                if process.stderr:
                    stderr = threading.Thread(target=consume, args=(process.stderr, saved), daemon=True)
                    stderr.start()
                consume(process.stdout, saved, kind == "live")
                if stderr:
                    stderr.join()
        except OSError as error:
            self._line("Log error: " + str(error))
        finally:
            code = process.wait()
            try:
                self.registry.reconcile(self.settings.path("runs_dir"))
            except (OSError, ValueError, sqlite3.Error) as error:
                self._line("Could not refresh results: " + str(error))
            with self.frames:
                process.stdout.close()
                if process.stderr:
                    process.stderr.close()
                if process.stdin:
                    process.stdin.close()
                self.job.update(status="complete" if code == 0 else "cancelled" if code == 130 else "failed", returncode=code)
                self.process = None
                self.live["ready"] = False
                self.frame = None
                self.revision += 1
                self.finished.set()
                self.frames.notify_all()

    def stop(self):
        with self.lock:
            if self.process is not None:
                write_bytes(self.cancel_file, b"stop\n")
                if self.job["kind"] == "live":
                    self.control({"type": "stop"})
                self.job["status"] = "stopping"
            return dict(self.job)

    def control(self, data):
        kind = choice(data, "type", ("capture", "finish", "target", "mode", "stop"), "capture")
        command = {"type": kind}
        if kind == "mode":
            command["mode"] = choice(data, "mode", ("automatic", "manual"), "automatic")
        elif kind == "target":
            target = text_field(data, "label")
            if target and target not in self.live.get("classes", []):
                raise ValueError("Choose an expected sign from the loaded model.")
            command["label"] = target or None
        with self.lock:
            if not self.process or self.job["kind"] != "live" or not self.process.stdin:
                raise ValueError("Start a live testing session first.")
            try:
                self.process.stdin.write(json.dumps(command) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise ValueError("The live session has ended.") from error

    def status(self, after=0):
        with self.lock:
            return {"job": dict(self.job), "logs": [entry for entry in self.logs if entry["sequence"] > after],
                    "sequence": self.sequence, "revision": self.revision, "live": dict(self.live),
                    "frame_version": self.frame_version, "simulator_url": self.simulator_url, "closing": self.closing}

    def record(self, identifier):
        record = next((r for r in self.registry.records() if r["id"] == identifier), None)
        if record is None:
            raise ValueError("The selected experiment was not found. Refresh the results.")
        return record

    def snapshot(self, variant="sequence", category="alphabet", identifier=""):
        choice({"version": variant}, "version", ("sequence", "static"), "sequence")
        records = self.registry.records()
        scan_path = self.root / "reports" / f"scan-{variant}.json"
        scan, rows, warnings = None, {}, []
        if scan_path.exists():
            try:
                scan = read_json(scan_path)
                rows = {row["category"]: row for row in scan["coverage"]}
                if not isinstance(scan["sources"], int) or not all(isinstance(row.get("signers"), list) and isinstance(row.get("signs"), list) for row in rows.values()):
                    raise ValueError("Invalid scan report")
            except (OSError, ValueError, KeyError, TypeError):
                scan, rows = None, {}
                warnings.append("The saved scan could not be read. Scan this input version again to rebuild its coverage report.")
        categories = [{"id": key, "name": name, "signs": len(self.catalog.selected(key, variant)),
                       "sources": rows.get(key, {}).get("sources", 0), "coverage": rows.get(key),
                       "target": len(self.catalog.selected(key, variant)) * len(self.settings.get("expected_signers")) * self.settings.get("takes_per_signer")}
                      for key, name in CATEGORY_NAMES.items()]
        matching = [r for r in records if r["kind"] in ("training", "training_fold") and r.get("variant") == variant and
                    (category == "all" or r.get("category") == category)]
        history_record = next((r for r in matching if (Path(r["path"]) / "history.json").is_file()), None)
        if identifier:
            selected = self.record(identifier)
            if (Path(selected["path"]) / "history.json").is_file():
                history_record = selected
        history = {}
        if history_record:
            try:
                value = read_json(Path(history_record["path"]) / "history.json")
                if isinstance(value, dict):
                    history = value
            except (OSError, ValueError):
                pass
        return {"root": str(self.root), "categories": categories, "records": records, "scan": scan,
                "history": history, "history_record": history_record, "tools": self.tools(), "initial": self.initial,
                "expected_signers": self.settings.get("expected_signers"), "takes_per_signer": self.settings.get("takes_per_signer"),
                "paths": {key: str(self.settings.path(key)) for key in ("raw_dir", "extracted_dir", "runs_dir", "exports_dir")},
                "revision": self.revision, "warnings": warnings}

    def result(self, identifier):
        record = self.record(identifier)
        directory = Path(record["path"])
        attempts = []
        truncated = False
        table = next((directory / name for name in ("attempts.csv", "predictions.csv", "trials.csv") if (directory / name).is_file()), None)
        if table:
            with table.open(newline="", encoding="utf-8-sig") as stream:
                for index, row in enumerate(csv.DictReader(stream)):
                    if index == 500:
                        truncated = True
                        break
                    attempts.append(row)
        files = sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix in (".png", ".csv", ".html")) if directory.is_dir() else []
        return {"record": record, "attempts": attempts, "truncated": truncated, "files": files,
                "history": read_json(directory / "history.json") if (directory / "history.json").is_file() else {}}

    def browse(self, value="", kind="directory"):
        directory = self.path(value) if value else self.root
        if directory.is_file():
            directory = directory.parent
        if not directory.is_dir():
            raise ValueError("This folder does not exist.")
        entries = []
        for path in directory.iterdir():
            try:
                is_directory = path.is_dir()
                if is_directory or (kind == "json" and path.suffix.lower() == ".json") or (kind == "python" and path.name.lower().startswith("python") and path.suffix.lower() == ".exe"):
                    entries.append({"name": path.name, "path": str(path), "directory": is_directory})
            except OSError:
                continue
        return {"path": str(directory), "parent": str(directory.parent), "entries": sorted(entries, key=lambda p: (not p["directory"], p["name"].casefold()))}

    def model_classes(self, value):
        directory = self.path(value)
        try:
            if (directory / "classes.json").is_file():
                classes = read_json(directory / "classes.json")
            elif (directory / "model.json").is_file():
                classes = read_json(directory / "model.json").get("classes", [])
            else:
                return []
            return classes if isinstance(classes, list) and all(isinstance(label, str) for label in classes) else []
        except (OSError, ValueError, AttributeError):
            return []

    def save_tools(self, data):
        with self.lock:
            if self.process is not None:
                raise ValueError("Wait for the current job before changing interpreter settings.")
            values = {key: text_field(data, key) for key in TOOL_KEYS}
            write_json(self.root / "settings.local.json", values)
            return values

    def open_path(self, value):
        path = self.path(value)
        if not path.exists():
            raise ValueError("This path has not been created yet.")
        allowed = [self.root, self.path(self.tools()["hudyat_path"])]
        if not path.is_dir() and (not any(path.is_relative_to(root) for root in allowed) or path.suffix.lower() not in (".log", ".txt", ".md", ".json", ".csv", ".png", ".html")):
            raise ValueError("Open a path in this workspace or the configured HUDYAT project.")
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, studio, port=0):
        self.studio = studio
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), StudioHandler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.cookie_name = f"hudyat_studio_{self.server_port}"

    def request_shutdown(self):
        with self.studio.lock:
            if self.studio.closing:
                return
            self.studio.closing = True
            self.studio.stop()
            process = self.studio.process
        def finish():
            if process:
                self.studio.finished.wait()
            self.shutdown()
        threading.Thread(target=finish, daemon=True).start()


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, value, status=200, content_type="application/json; charset=utf-8", extra=None):
        if not isinstance(value, bytes):
            value = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(value)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
        for key, header_value in (extra or {}).items():
            self.send_header(key, str(header_value))
        self.end_headers()
        self.wfile.write(value)

    def _check_host(self):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            raise PermissionError("Use the trainer's 127.0.0.1 address.")

    def _check_session(self):
        cookie = f"{self.server.cookie_name}={self.server.token}"
        if cookie not in [part.strip() for part in self.headers.get("Cookie", "").split(";")]:
            raise PermissionError("Open the trainer's home page to start a local session.")

    def do_GET(self):
        try:
            self._check_host()
            url = urlsplit(self.path)
            query = parse_qs(url.query, keep_blank_values=True)
            get = lambda key, default="": query.get(key, [default])[0]
            if url.path in ("/", "/index.html"):
                html = (ASSETS / "index.html").read_text(encoding="utf-8").replace("__STUDIO_TOKEN__", self.server.token)
                self._send(html.encode(), content_type="text/html; charset=utf-8",
                           extra={"Set-Cookie": f"{self.server.cookie_name}={self.server.token}; HttpOnly; SameSite=Strict; Path=/"})
                return
            if url.path in ("/app.js", "/icons.js", "/style.css", "/studio-extra.css"):
                path = ASSETS / url.path[1:]
                self._send(path.read_bytes(), content_type="text/javascript; charset=utf-8" if path.suffix == ".js" else "text/css; charset=utf-8")
                return
            self._check_session()
            studio = self.server.studio
            if url.path == "/api/state":
                self._send(studio.snapshot(get("version", "sequence"), get("category", "alphabet"), get("run")))
            elif url.path == "/api/status":
                self._send(studio.status(int(get("after", "0"))))
            elif url.path == "/api/results":
                self._send(studio.result(get("id")))
            elif url.path == "/api/browse":
                self._send(studio.browse(get("path"), get("kind", "directory")))
            elif url.path == "/api/model":
                self._send({"classes": studio.model_classes(get("path"))})
            elif url.path == "/api/frame":
                after = int(get("after", "0"))
                with studio.frames:
                    studio.frames.wait_for(lambda: studio.frame_version > after or studio.process is None, timeout=1)
                    frame, version = studio.frame, studio.frame_version
                self._send(frame or b"", status=200 if frame else 204, content_type="image/png", extra={"X-Frame-Version": version})
            elif url.path.startswith("/artifacts/"):
                parts = url.path.split("/", 3)
                if len(parts) != 4:
                    raise ValueError("Choose a report file.")
                record = studio.record(unquote(parts[2]))
                directory = Path(record["path"]).resolve()
                path = (directory / unquote(parts[3])).resolve()
                if not directory.is_relative_to(studio.root) or not path.is_relative_to(directory):
                    raise PermissionError("The report path is outside this workspace.")
                if path.suffix.lower() not in (".html", ".csv", ".png", ".json") or not path.is_file():
                    raise ValueError("This report artifact is not available.")
                self._send(path.read_bytes(), content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            else:
                self._send({"error": "Not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except PermissionError as error:
            self._send({"error": str(error)}, 403)
        except (ValueError, KeyError, OSError) as error:
            self._send({"error": str(error)}, 400)
        except Exception as error:
            self._send({"error": "Trainer error: " + str(error)}, 500)

    def do_POST(self):
        try:
            self._check_host()
            self._check_session()
            if self.headers.get("Origin") != self.server.origin or not secrets.compare_digest(self.headers.get("X-Studio-Token", ""), self.server.token):
                raise PermissionError("Requests must come from this trainer window.")
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                raise ValueError("Send a JSON request.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024 * 1024:
                raise ValueError("Invalid request size.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            studio = self.server.studio
            route = urlsplit(self.path).path
            if route == "/api/jobs":
                self._send(studio.start(text_field(data, "action"), data), 202)
            elif route == "/api/stop":
                self._send(studio.stop())
            elif route == "/api/live":
                studio.control(data)
                self._send({"ok": True})
            elif route == "/api/settings":
                self._send(studio.save_tools(data))
            elif route == "/api/open":
                studio.open_path(text_field(data, "path"))
                self._send({"ok": True})
            elif route == "/api/refresh":
                studio.registry.reconcile(studio.settings.path("runs_dir"))
                studio.revision += 1
                self._send({"ok": True})
            elif route == "/api/shutdown":
                self._send({"closing": True}, 202)
                self.server.request_shutdown()
            else:
                self._send({"error": "Not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except PermissionError as error:
            self._send({"error": str(error)}, 403)
        except (ValueError, KeyError, OSError) as error:
            self._send({"error": str(error)}, 400)
        except Exception as error:
            self._send({"error": "Trainer error: " + str(error)}, 500)


def main(root: Path = ROOT, testing_run=None, camera=0, mode="automatic", port=0, open_browser=True):
    studio = Studio(root, testing_run, camera, mode)
    server = StudioServer(studio, port)
    write_bytes(studio.root / "tmp" / "studio-url.txt", (server.origin + "/\n").encode())
    if sys.stdout:
        print("HUDYAT Studio: " + server.origin + "/", flush=True)
    if open_browser:
        webbrowser.open(server.origin + "/")
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        studio.stop()
    finally:
        studio.stop()
        process = studio.process
        if process:
            studio.finished.wait()
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HUDYAT local browser trainer")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    options = parser.parse_args()
    main(options.root, port=options.port, open_browser=not options.no_browser)
