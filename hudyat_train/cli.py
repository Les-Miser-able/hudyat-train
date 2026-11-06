from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys

from .catalog import Catalog, CATEGORY_NAMES
from .config import ROOT, VARIANTS, load_settings
from .jobs import Cancelled, Job
from .storage import write_json


def initialize(settings, folders=True):
    for key in ("raw_dir", "extracted_dir", "cache_dir", "runs_dir", "exports_dir"):
        settings.path(key).mkdir(parents=True, exist_ok=True)
    if folders:
        for sign in Catalog(settings.path("signs_file")).signs:
            for signer in settings.get("expected_signers"):
                (settings.path("raw_dir") / sign.category / sign.sign_id / signer).mkdir(parents=True, exist_ok=True)


def parser():
    p = argparse.ArgumentParser(description="HUDYAT local extraction, training, and compatible model exports.")
    p.add_argument("--root", type=Path, default=ROOT, help="Project directory containing project.json")
    p.add_argument("--cancel-file", type=Path, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("gui", help="Open the local browser studio")
    sub.add_parser("init", help="Create category/class/signer folders without changing raw files")
    sub.add_parser("doctor", help="Check Python packages, detector assets, and local paths")
    audit = sub.add_parser("audit", help="Compare fixed study signs and settings with HUDYAT's checked-in code")
    audit.add_argument("--hudyat", type=Path, default=ROOT.parent / "HUDYAT")
    scan = sub.add_parser("scan", help="Scan raw data and show vocabulary/signer coverage")
    scan.add_argument("--category")
    scan.add_argument("--variant", choices=VARIANTS, default="sequence")
    scan.add_argument("--report", type=Path)
    extract = sub.add_parser("extract", help="Extract and quality-check a category")
    extract.add_argument("--category", required=True, help="A category ID, or 'all' for categories containing raw files")
    extract.add_argument("--variant", choices=(*VARIANTS, "both"), default="sequence")
    training = sub.add_parser("train", help="Train and evaluate a category from extracted samples")
    training.add_argument("--category", required=True, choices=list(CATEGORY_NAMES))
    training.add_argument("--variant", choices=VARIANTS, default="sequence")
    training.add_argument("--protocol", choices=("random", "loso"), default="random")
    training.add_argument("--epochs", type=int, default=150)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--augmentation-copies", type=int, default=5,
                          help="Training-only augmented copies per original (default: 5; 0 disables; other counts require --development)")
    training.add_argument("--mirroring", choices=("all", "approved", "off"), default="all",
                          help="Left/right training augmentation: all signs (default), only mirror_ok=1 signs, or off")
    training.add_argument("--development", action="store_true", help="Explicitly allow incomplete study data or fewer epochs")
    exporting = sub.add_parser("export", help="Stage a HUDYAT catalog, model metadata, weights, and detectors")
    exporting.add_argument("--run", required=True, type=Path)
    exporting.add_argument("--output", type=Path, help="Defaults to exports/<variant>; preserves other categories in the catalog")
    exporting.add_argument("--allow-development", action="store_true")
    verify = sub.add_parser("verify", help="Check exported predictions and preprocessing against HUDYAT's own code")
    verify.add_argument("--hudyat", type=Path, default=ROOT.parent / "HUDYAT")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--tfjs", type=Path, help="Optional installed @tensorflow/tfjs package directory")
    testing = sub.add_parser("test", help="Automatically replay an evaluated run's held-out test samples")
    testing.add_argument("--run", type=Path, required=True)
    results = sub.add_parser("results", help="Refresh the experiment history and its CSV comparison table")
    results.add_argument("--rebuild", action="store_true", help="Also generate graphs and HTML reports for older completed runs")
    live = sub.add_parser("live", help="Open the browser studio's live model tester")
    live.add_argument("--run", type=Path, required=True)
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--mode", choices=("automatic", "manual"), default="automatic")
    worker = sub.add_parser("live-worker", help="Internal desktop camera worker / reproducible video replay")
    worker.add_argument("--run", type=Path, required=True)
    worker.add_argument("--camera", type=int, default=0)
    worker.add_argument("--mode", choices=("automatic", "manual"), default="automatic")
    worker.add_argument("--target")
    worker.add_argument("--video", type=Path)
    worker.add_argument("--control-stdin", action="store_true")
    worker.add_argument("--no-preview", action="store_true")
    worker.add_argument("--max-attempts", type=int)
    simulation = sub.add_parser("simulate", help="Run HUDYAT's actual browser camera, settings, and recognition code with a selected model")
    simulation.add_argument("--run", type=Path, required=True)
    simulation.add_argument("--hudyat", type=Path, default=ROOT.parent / "HUDYAT")
    simulation.add_argument("--candidate", type=Path, help="Optional Alphabet candidate run/export for the Recognition lab")
    simulation.add_argument("--profile", choices=("baseline", "enhanced"), default="baseline")
    simulation.add_argument("--settings-file", type=Path, help="Settings or Copy results JSON from HUDYAT's Recognition lab")
    simulation.add_argument("--target")
    simulation.add_argument("--port", type=int, default=0, help="Local port; 0 selects an available port")
    simulation.add_argument("--no-browser", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        settings = load_settings(args.root)
        def project_path(path):
            return (settings.root / path).resolve() if not path.is_absolute() else path.resolve()
        job = Job(args.cancel_file)
        job.check()
        if args.command == "gui":
            from .web_app import main as gui
            gui(settings.root)
        elif args.command == "init":
            initialize(settings)
            job.emit(f"Raw data folders ready: {settings.path('raw_dir')}")
        elif args.command == "doctor":
            for package in ("numpy", "mediapipe", "opencv-contrib-python", "tensorflow", "Pillow", "matplotlib"):
                try:
                    version = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    version = "MISSING - run setup.ps1"
                job.emit(f"{package}: {version}")
            for key in ("hand_model", "face_model", "raw_dir"):
                path = settings.path(key)
                job.emit(f"{key}: {path} ({'found' if path.exists() else 'missing'})")
        elif args.command == "audit":
            from .integration import audit
            report = audit(settings, args.hudyat.resolve(), job.emit)
            write_json(settings.root / "reports" / "integration-audit.json", report)
        elif args.command == "scan":
            from .dataset import scan
            report = scan(settings, args.category, args.variant, job.emit)
            write_json(project_path(args.report) if args.report else settings.root / "reports" / f"scan-{args.variant}.json", report)
            if report["issues"]:
                return 2
        elif args.command == "extract":
            from .dataset import discover
            from .extraction import extract
            variants = VARIANTS if args.variant == "both" else (args.variant,)
            done = 0
            if args.category == "all":
                _, issues, _ = discover(settings, variant="sequence")
                if issues:
                    raise ValueError("Fix dataset scan errors before extracting all categories.")
            for variant in variants:
                categories = list(CATEGORY_NAMES) if args.category == "all" else [args.category]
                for category in categories:
                    if args.category == "all" and not discover(settings, category, variant)[0]:
                        continue
                    extract(settings, category, variant, job)
                    done += 1
            if not done:
                raise ValueError("No eligible raw data found. Add videos/images in dataset/raw first.")
        elif args.command == "train":
            from .training import TrainOptions, train
            result = train(settings, TrainOptions(args.category, args.variant, args.protocol, args.epochs,
                                                  augmentation_copies=args.augmentation_copies, seed=args.seed, development=args.development,
                                                  mirroring=args.mirroring), job)
            job.emit(f"RUN_PATH={result}")
        elif args.command == "export":
            from .exporting import export_run
            export_run(settings, project_path(args.run), project_path(args.output) if args.output else None, args.allow_development, job.emit)
        elif args.command == "test":
            from .testing import automated_test
            automated_test(settings, project_path(args.run), job)
        elif args.command == "results":
            from .experiments import Registry
            records = Registry(settings.root).reconcile(settings.path("runs_dir"))
            if args.rebuild:
                from .reporting import loso_report, result_report
                from .storage import read_json
                for record in records:
                    job.check()
                    path = Path(record["path"])
                    if record.get("status") != "complete":
                        continue
                    if (path / "loso-summary.json").exists():
                        loso_report(path, read_json(path / "loso-summary.json"), record)
                    else:
                        attempts = read_json(path / "attempts.json") if (path / "attempts.json").exists() else None
                        result_report(path, record, record["metrics"], attempts)
            for record in records:
                score = record["metrics"].get("accuracy")
                accuracy = f"{score:.1%}" if score is not None else "unscored"
                job.emit(f"{record['kind']} | {record.get('category')} | {record.get('variant')} | {record.get('status')} | {accuracy} | {record['path']}")
            job.emit(f"Experiment table: {settings.root / 'reports' / 'experiments.csv'}")
        elif args.command == "live":
            from .web_app import main as gui
            gui(settings.root, testing_run=project_path(args.run), camera=args.camera, mode=args.mode)
        elif args.command == "live-worker":
            from .realtime import run_live
            if args.max_attempts is not None and args.max_attempts < 1:
                raise ValueError("--max-attempts must be positive.")
            run_live(settings, project_path(args.run), args.camera, args.mode, args.target,
                     project_path(args.video) if args.video else None, args.control_stdin, args.max_attempts, job, preview=not args.no_preview)
        elif args.command == "simulate":
            from .simulation import run_simulator
            run_simulator(settings, project_path(args.run), args.hudyat.resolve(),
                          project_path(args.candidate) if args.candidate else None, args.profile,
                          project_path(args.settings_file) if args.settings_file else None, args.target,
                          args.port, not args.no_browser, job)
        elif args.command == "verify":
            from .parity import write_fixture
            fixture = write_fixture(settings.root / "tmp" / "browser-preprocessing-parity.json")
            command = ["node", "--experimental-strip-types", str(ROOT / "tools" / "verify_hudyat.mjs"),
                       "--hudyat", str(args.hudyat.resolve()), "--bundle", str(project_path(args.bundle)), "--fixture", str(fixture)]
            if args.tfjs:
                command.extend(["--tfjs", str(args.tfjs.resolve())])
            return subprocess.run(command, cwd=settings.root, check=False).returncode
    except Cancelled as error:
        print(str(error), flush=True)
        return 130
    except (ValueError, OSError, KeyError, importlib.metadata.PackageNotFoundError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    return 0
