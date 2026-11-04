"""Native desktop workflow. Jobs run in subprocesses; Tk stays responsive."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from time import monotonic


def _prepare_tcl():
    # This must precede importing _tkinter, which caches Tcl's library location.
    if os.name == "nt":
        import ctypes
        library = Path(sys.base_prefix) / "DLLs" / "tcl86t.dll"
        if library.is_file():
            tcl = ctypes.CDLL(str(library))
            tcl.Tcl_FindExecutable.argtypes = [ctypes.c_char_p]
            tcl.Tcl_FindExecutable.restype = None
            tcl.Tcl_FindExecutable(sys.executable.encode("utf-8"))


_prepare_tcl()
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid
import webbrowser

from .catalog import CATEGORY_NAMES
from .cli import initialize
from .config import ROOT, load_settings
from .storage import read_json, write_bytes, write_json

MIRRORING_POLICIES = {"All signs": "all", "Approved signs only": "approved", "Off": "off"}


class App:
    def __init__(self, window: tk.Tk, root: Path = ROOT):
        self.window, self.settings = window, load_settings(root)
        initialize(self.settings)
        self.root = self.settings.root
        self.events, self.process, self.closing = queue.Queue(), None, False
        self.cancel_file, self.scan_report, self.job_kind = None, None, ""
        self.actions = []
        self.category = tk.StringVar(value="alphabet")
        self.variant = tk.StringVar(value="sequence")
        self.protocol = tk.StringVar(value="random")
        self.epochs = tk.StringVar(value="150")
        self.seed = tk.StringVar(value=str(self.settings.get("seed")))
        self.augmentation_enabled = tk.BooleanVar(value=True)
        self.augmentation_copies = tk.StringVar(value="5")
        self.mirroring = tk.StringVar(value="All signs")
        self.augmentation_summary = tk.StringVar()
        self.development = tk.BooleanVar(value=False)
        self.allow_development = tk.BooleanVar(value=False)
        self.run_path = tk.StringVar()
        self.test_run = tk.StringVar()
        self.live_camera = tk.StringVar(value="0")
        self.live_mode = tk.StringVar(value="automatic")
        self.live_target = tk.StringVar(value="(unscored)")
        self.simulation_profile = tk.StringVar(value="baseline")
        self.simulation_candidate = tk.StringVar()
        self.simulation_settings = tk.StringVar()
        self.simulation_url = tk.StringVar()
        self.live_prediction = tk.StringVar(value="No prediction yet")
        self.live_feedback = tk.StringVar(value="Choose a model and start the camera.")
        self.live_diagnostics = tk.StringVar()
        self.live_frame, self.live_image, self.live_ready = None, None, False
        self.live_frame_lock = threading.Lock()
        self.last_scored_at = -10.0
        self.export_path = tk.StringVar(value=str(self.settings.path("exports_dir") / "sequence"))
        self.status = tk.StringVar(value="Ready. Add raw data, then scan the dataset.")
        self.coverage_status = tk.StringVar(value="Drop videos or images into the category/class/signer folders, then click Scan dataset.")
        local = read_json(self.root / "settings.local.json") if (self.root / "settings.local.json").exists() else {}
        self.extract_python = tk.StringVar(value=local.get("extraction_python", ""))
        self.train_python = tk.StringVar(value=local.get("training_python", ""))
        self.hudyat_path = tk.StringVar(value=local.get("hudyat_path", str(self.root.parent / "HUDYAT")))
        self.tfjs_path = tk.StringVar(value=local.get("tfjs_path", ""))
        window.title("HUDYAT Train")
        window.geometry("1180x860")
        window.minsize(1040, 760)
        window.configure(bg="#f6f5ef")
        style = ttk.Style(window)
        style.theme_use("clam")
        style.configure("TFrame", background="#f6f5ef")
        style.configure("TLabel", background="#f6f5ef", foreground="#202522", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 25, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Accent.TButton", background="#f2bf42", foreground="#202522", padding=(16, 9), font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("disabled", "#d8d6cf"), ("active", "#f5cd6e")])
        style.configure("TButton", padding=(10, 6), font=("Segoe UI", 10))
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        shell = ttk.Frame(window, padding=20)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="HUDYAT Train", style="Title.TLabel").pack(anchor="w")
        ttk.Label(shell, text="Raw data  /  Landmark extraction  /  Conv1D-BiLSTM training  /  HUDYAT export").pack(anchor="w", pady=(3, 14))
        selection = ttk.Frame(shell)
        selection.pack(fill="x", pady=(0, 14))
        ttk.Label(selection, text="Category").pack(side="left")
        self.category_box = ttk.Combobox(selection, textvariable=self.category, values=["all", *CATEGORY_NAMES], width=22, state="readonly")
        self.category_box.pack(side="left", padx=(8, 22))
        ttk.Label(selection, text="Version").pack(side="left")
        self.variant_box = ttk.Combobox(selection, textvariable=self.variant, values=["sequence", "static"], width=15, state="readonly")
        self.variant_box.pack(side="left", padx=8)
        self.variant_box.bind("<<ComboboxSelected>>", self._variant_changed)
        ttk.Label(selection, text="sequence = static + movement   |   static = held poses").pack(side="left", padx=12)
        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill="both", expand=True)
        data, extraction, training, testing, results, exporting, settings = [self._tab(name) for name in
                                                        ("Dataset", "Extract", "Train", "Test models", "Results", "Export", "Settings")]
        self.testing_tab = testing
        self._dataset_page(data)
        self._extraction_page(extraction)
        self._training_page(training)
        self._testing_page(testing)
        from .results_view import ResultsView
        self.results_view = ResultsView(results, self.settings, self.open_path, self._start, self.choose_test_model)
        self.results_view.pack(fill="both", expand=True)
        self._export_page(exporting)
        self._settings_page(settings)
        footer = ttk.Frame(shell)
        footer.pack(fill="x", pady=(12, 5))
        ttk.Label(footer, textvariable=self.status).pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(footer, text="Stop job", command=self.stop, state="disabled")
        self.stop_button.pack(side="right")
        self.progress = ttk.Progressbar(shell, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 6))
        console = ttk.Frame(shell)
        console.pack(fill="x")
        self.log = tk.Text(console, height=8, wrap="word", bg="#202923", fg="#edf1eb", insertbackground="white",
                           font=("Consolas", 9), relief="flat", padx=10, pady=8, state="disabled")
        scrollbar = ttk.Scrollbar(console, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._append("Workspace: " + str(self.root))
        self._append("Raw media stays in place. Extraction and training write derived artifacts into this project.")
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.bind("<space>", lambda event: self._capture_key(event, "capture"))
        window.bind("<Return>", lambda event: self._capture_key(event, "finish"))
        window.after(100, self._poll)

    def _tab(self, name):
        frame = ttk.Frame(self.tabs, padding=16)
        self.tabs.add(frame, text=name)
        return frame

    def _heading(self, frame, title, text):
        ttk.Label(frame, text=title, style="Heading.TLabel").pack(anchor="w")
        ttk.Label(frame, text=text, wraplength=1030, justify="left").pack(anchor="w", pady=(5, 12))

    def _button(self, frame, text, command, accent=False):
        button = ttk.Button(frame, text=text, command=command, style="Accent.TButton" if accent else "TButton")
        self.actions.append(button)
        return button

    def _path_row(self, parent, label, variable, browse=None):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text=label, width=20).pack(side="left")
        ttk.Entry(frame, textvariable=variable).pack(side="left", fill="x", expand=True)
        if browse:
            ttk.Button(frame, text="Browse", command=browse).pack(side="right", padx=(8, 0))

    def _dataset_page(self, frame):
        self._heading(frame, "Your local raw dataset", "Place media under dataset/raw/<category>/<class>/<signer>/. Videos and images can also be inside subfolders such as video/ or image/. Class IDs and display labels are accepted.")
        tools = ttk.Frame(frame)
        tools.pack(fill="x", pady=(0, 10))
        self._button(tools, "Scan dataset", self.scan, True).pack(side="left")
        ttk.Button(tools, text="Open raw data folder", command=lambda: self.open_path(self.settings.path("raw_dir"))).pack(side="left", padx=8)
        ttk.Label(frame, textvariable=self.coverage_status, wraplength=1030).pack(anchor="w", pady=(0, 8))
        columns = ("category", "files", "classes", "signers")
        self.coverage = ttk.Treeview(frame, columns=columns, show="headings", height=5, selectmode="browse")
        for name, title, width in [("category", "Category", 170), ("files", "Raw files", 95), ("classes", "Classes", 95), ("signers", "Signers", 350)]:
            self.coverage.heading(name, text=title)
            self.coverage.column(name, width=width, anchor="w")
        self.coverage.pack(fill="x")
        self.coverage.bind("<<TreeviewSelect>>", self._coverage_selected)
        ttk.Label(frame, text="Selected category: raw files per sign and signer (target: 15 per cell)").pack(anchor="w", pady=(10, 4))
        details = ttk.Frame(frame)
        details.pack(fill="both", expand=True)
        self.sign_counts = ttk.Treeview(details, columns=("sign", *self.settings.get("expected_signers")), show="headings", height=5)
        self.sign_counts.heading("sign", text="Sign")
        self.sign_counts.column("sign", width=225)
        for signer in self.settings.get("expected_signers"):
            self.sign_counts.heading(signer, text=signer)
            self.sign_counts.column(signer, width=75, anchor="center")
        scroll = ttk.Scrollbar(details, command=self.sign_counts.yview)
        self.sign_counts.configure(yscrollcommand=scroll.set)
        self.sign_counts.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _extraction_page(self, frame):
        self._heading(frame, "Extract model inputs", "Decode the original media, find hand landmarks, trim to the sign window, and measure detection quality inside that window. Accepted samples are saved as 32-frame NumPy files; rejected sources stay on disk with a reason in the index.")
        for text in ["Sequence: preserves movement and supports both static and moving signs.",
                     "Static: chooses one stable pose per source and repeats it. Classes marked dynamic (currently J and Z) are excluded.",
                     "Family: includes 13 hand-to-face location values, for 141 features instead of 128.",
                     "Repeating extraction reuses cached detections and rebuilds the current source index."]:
            ttk.Label(frame, text=text, wraplength=1000).pack(anchor="w", pady=8)
        self._button(frame, "Extract selected category / version", self.extract, True).pack(anchor="w", pady=18)
        ttk.Button(frame, text="Open extracted samples", command=lambda: self.open_path(self.settings.path("extracted_dir"))).pack(anchor="w")

    def _training_page(self, frame):
        viewport = tk.Canvas(frame, background="#f6f5ef", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=viewport.yview)
        viewport.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        viewport.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(viewport)
        content_window = viewport.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda event: viewport.configure(scrollregion=viewport.bbox("all")))
        viewport.bind("<Configure>", lambda event: viewport.itemconfigure(content_window, width=event.width))
        frame = content
        self._heading(frame, "Train and evaluate", "One Conv1D-BiLSTM model is trained per category and version. Splits are made before augmentation. Validation and test each contain one unmodified sample per original source.")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text="Evaluation protocol", width=23).pack(side="left")
        ttk.Combobox(row, textvariable=self.protocol, values=["random", "loso"], state="readonly", width=20).pack(side="left")
        ttk.Label(frame, text="random: per-class 70/15/15 split.   loso: each signer is tested once, with another signer used for validation.", wraplength=1000).pack(anchor="w", pady=8)
        fields = ttk.Frame(frame)
        fields.pack(fill="x", pady=8)
        ttk.Label(fields, text="Maximum epochs").pack(side="left")
        ttk.Entry(fields, textvariable=self.epochs, width=8).pack(side="left", padx=(10, 25))
        ttk.Label(fields, text="Random seed").pack(side="left")
        ttk.Entry(fields, textvariable=self.seed, width=8).pack(side="left", padx=10)
        ttk.Label(frame, text="Study defaults: Adam 0.001, batch size 32, up to 150 epochs, early stopping patience 20, and five training-only augmented copies.", wraplength=1000).pack(anchor="w", pady=8)
        augmentation = ttk.LabelFrame(frame, text="Data augmentation", padding=10)
        augmentation.pack(fill="x", pady=6)
        controls = ttk.Frame(augmentation)
        controls.pack(fill="x")
        ttk.Checkbutton(controls, text="Enable training augmentation", variable=self.augmentation_enabled,
                        command=self._augmentation_controls).pack(side="left")
        ttk.Label(controls, text="Augmented copies per original").pack(side="left", padx=(20, 8))
        self.augmentation_box = ttk.Spinbox(controls, from_=1, to=100, textvariable=self.augmentation_copies, width=5)
        self.augmentation_box.pack(side="left")
        mirroring = ttk.Frame(augmentation)
        mirroring.pack(fill="x", pady=(8, 0))
        ttk.Label(mirroring, text="Left/right mirroring").pack(side="left")
        self.mirroring_box = ttk.Combobox(mirroring, textvariable=self.mirroring, values=list(MIRRORING_POLICIES),
                                         state="readonly", width=23)
        self.mirroring_box.pack(side="left", padx=8)
        ttk.Label(augmentation, textvariable=self.augmentation_summary, wraplength=980).pack(anchor="w", pady=(6, 0))
        ttk.Label(augmentation, text="Mirroring swaps left/right hands and reflects horizontal movement and face locations to teach either hand. Approved signs only uses mirror_ok in signs.csv. Other copies use noise, scaling and rotation; sequences also use timing changes and frame removal.", wraplength=980).pack(anchor="w", pady=(4, 0))
        self.augmentation_copies.trace_add("write", self._augmentation_controls)
        self.mirroring.trace_add("write", self._augmentation_controls)
        self._augmentation_controls()
        ttk.Checkbutton(frame, text="Development run: allow incomplete study data or fewer epochs", variable=self.development).pack(anchor="w", pady=8)
        ttk.Label(frame, text="Study runs require all expected classes and at least 15 accepted takes for each of S1-S7. Development results are explicitly marked and need a separate export choice.", wraplength=1000).pack(anchor="w", pady=5)
        self._button(frame, "Train selected category / version", self.train, True).pack(anchor="w", pady=15)
        ttk.Button(frame, text="Open runs and evaluation reports", command=lambda: self.open_path(self.settings.path("runs_dir"))).pack(anchor="w")

    def _augmentation_controls(self, *args):
        enabled = self.augmentation_enabled.get()
        self.augmentation_box.configure(state="normal" if enabled else "disabled")
        self.mirroring_box.configure(state="readonly" if enabled else "disabled")
        try:
            copies = int(self.augmentation_copies.get()) if enabled else 0
            if copies < 1 and enabled:
                raise ValueError()
        except ValueError:
            self.augmentation_summary.set("Enter a positive whole number of augmented copies.")
            return
        if not copies or self.mirroring.get() == "Off":
            mirror_summary = "Left/right mirroring is off."
        else:
            scope = "per original" if self.mirroring.get() == "All signs" else "per approved original"
            mirror_summary = f"Includes {(copies + 1) // 2} mirrored copies {scope}, starting with an exact reflection."
        self.augmentation_summary.set(f"Each original produces {copies} augmented copies ({copies + 1} training samples total). {mirror_summary} Validation and test samples stay unmodified. Study default: 5 copies.")

    def _export_page(self, frame):
        self._heading(frame, "Export for HUDYAT", "Choose a completed random-split run. The export contains HUDYAT's existing model.json + weights.bin format, a catalog, matching detector assets, and synthetic parity probes. Static and sequence releases use separate output folders.")
        self._path_row(frame, "Training run", self.run_path, lambda: self.browse_dir(self.run_path, self.settings.path("runs_dir")))
        self._path_row(frame, "Bundle directory", self.export_path, lambda: self.browse_dir(self.export_path, self.settings.path("exports_dir")))
        ttk.Checkbutton(frame, text="Allow export of a development model for testing", variable=self.allow_development).pack(anchor="w", pady=10)
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=8)
        self._button(row, "Export model bundle", self.export, True).pack(side="left")
        self._button(row, "Verify against HUDYAT", self.verify).pack(side="left", padx=10)
        ttk.Button(row, text="Open bundle folder", command=lambda: self.open_path(Path(self.export_path.get()))).pack(side="left")
        ttk.Label(frame, text="Verification loads the exported weights with HUDYAT's own TensorFlow.js loader and compares predictions and preprocessing with Python. It needs Node.js and HUDYAT's installed client dependencies.", wraplength=1000).pack(anchor="w", pady=12)
        ttk.Label(frame, text="This stages local files. It does not register models in HUDYAT's database. See README.md for installation and registration details.", wraplength=1000).pack(anchor="w", pady=5)
        self._button(frame, "Check HUDYAT lesson compatibility", lambda: self._start(["audit", "--hudyat", self.hudyat_path.get()], "audit")).pack(anchor="w", pady=8)

    def _testing_page(self, frame):
        self._heading(frame, "Automated and manual model testing", "Select an evaluated run, a LOSO fold, or an exported model version. Testing automatically saves its results, CSV tables, graphs, and an HTML report.")
        self._path_row(frame, "Model / training run", self.test_run, self.choose_test_model)
        self.test_modes = ttk.Notebook(frame)
        self.test_modes.pack(fill="both", expand=True, pady=8)
        automatic, live, simulation = [ttk.Frame(self.test_modes, padding=10) for _ in range(3)]
        self.test_modes.add(automatic, text="Automated held-out test")
        self.test_modes.add(live, text="Live webcam / manual test")
        self.test_modes.add(simulation, text="Simulate HUDYAT")
        self._simulation_page(simulation)
        ttk.Label(automatic, text="Replays the exact saved test split. A LOSO parent replays every held-out fold. Original labels and sample checksums are checked; warm-up is excluded from timing.", wraplength=950).pack(anchor="w", pady=10)
        self._button(automatic, "Run automated test", self.automated_test, True).pack(anchor="w", pady=10)
        ttk.Label(automatic, text="Results include raw top-1 accuracy, per-class precision/recall/F1, rejection counts at the model's confidence threshold, and Python model latency. This does not measure HUDYAT browser response time.", wraplength=950).pack(anchor="w", pady=10)
        controls = ttk.Frame(live)
        controls.pack(fill="x")
        ttk.Label(controls, text="Camera index").pack(side="left")
        ttk.Entry(controls, textvariable=self.live_camera, width=4).pack(side="left", padx=(5, 12))
        self._button(controls, "Start camera", self.start_live, True).pack(side="left")
        self.live_stop = ttk.Button(controls, text="Stop & save reports", command=lambda: self._send_live({"type": "stop"}), state="disabled")
        self.live_stop.pack(side="left", padx=6)
        for value, label in [("automatic", "Automatic sign window"), ("manual", "Manual capture")]:
            ttk.Radiobutton(controls, text=label, value=value, variable=self.live_mode, command=self._live_mode_changed).pack(side="left", padx=5)
        target = ttk.Frame(live)
        target.pack(fill="x", pady=7)
        ttk.Label(target, text="Expected sign").pack(side="left")
        self.target_box = ttk.Combobox(target, textvariable=self.live_target, values=["(unscored)"], state="readonly", width=23)
        self.target_box.pack(side="left", padx=8)
        self.target_box.bind("<<ComboboxSelected>>", lambda _: self._send_live({"type": "target", "label": self._target()}))
        self.capture_button = ttk.Button(target, text="Start capture (Space)", command=lambda: self._send_live({"type": "capture"}), state="disabled")
        self.capture_button.pack(side="left", padx=5)
        self.finish_button = ttk.Button(target, text="Finish & evaluate (Enter)", command=lambda: self._send_live({"type": "finish"}), state="disabled")
        self.finish_button.pack(side="left", padx=5)
        view = ttk.Frame(live)
        view.pack(fill="both", expand=True)
        self.camera_preview = ttk.Label(view, text="Live camera preview", anchor="center", width=54)
        self.camera_preview.pack(side="left", fill="both", expand=True)
        prediction = ttk.Frame(view, padding=(15, 0))
        prediction.pack(side="right", fill="both", expand=True)
        ttk.Label(prediction, textvariable=self.live_prediction, font=("Segoe UI", 17, "bold"), wraplength=420).pack(anchor="w", pady=5)
        self.feedback_label = ttk.Label(prediction, textvariable=self.live_feedback, wraplength=430)
        self.feedback_label.pack(anchor="w", pady=5)
        ttk.Label(prediction, textvariable=self.live_diagnostics, wraplength=420).pack(anchor="w", pady=5)
        self.top_predictions = ttk.Treeview(prediction, columns=("label", "confidence"), show="headings", height=3)
        self.top_predictions.heading("label", text="Top predictions")
        self.top_predictions.heading("confidence", text="Confidence %")
        self.top_predictions.column("label", width=220)
        self.top_predictions.column("confidence", width=110)
        self.top_predictions.pack(fill="x", pady=5)

    def _simulation_page(self, frame):
        self._heading(frame, "Test with HUDYAT's camera and recognition", "Opens a local browser simulator using the current HUDYAT project's actual camera, model loader, Recognition lab and decision rules. The selected model stays local.")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="Initial HUDYAT profile").pack(side="left")
        ttk.Combobox(row, textvariable=self.simulation_profile, values=["baseline", "enhanced"], state="readonly", width=15).pack(side="left", padx=10)
        ttk.Label(frame, text="The browser includes all HUDYAT settings: resolution, CPU/GPU, hand tracking, background blur/removal, sign windows, early checks, hold confirmation, confidence and margin. You can import custom settings copied from HUDYAT.", wraplength=950).pack(anchor="w", pady=8)
        self._path_row(frame, "Alphabet candidate", self.simulation_candidate,
                       lambda: self.browse_dir(self.simulation_candidate, self.settings.path("runs_dir")))
        self._path_row(frame, "Settings JSON (optional)", self.simulation_settings, self._browse_simulation_settings)
        ttk.Label(frame, text="Choose a completed random-split run or an exported model version above. Candidate comparison is optional and applies to Alphabet. The HUDYAT source folder is selected in Settings. Run setup-simulator.cmd once to install the browser tools.", wraplength=950).pack(anchor="w", pady=8)
        self._button(frame, "Open HUDYAT camera simulator", self.start_simulation, True).pack(anchor="w", pady=12)
        self._path_row(frame, "Local simulator URL", self.simulation_url,
                       lambda: webbrowser.open(self.simulation_url.get()) if self.simulation_url.get().startswith("http://127.0.0.1:") else None)
        ttk.Label(frame, text="Allow the camera in the browser, choose an expected sign, and enter the camera step. Browser camera selection replaces the native tester's camera index. Results save automatically by settings configuration and appear in Results; camera pixels and landmark arrays are not saved. Finish in the browser or use Stop job here.", wraplength=950).pack(anchor="w", pady=8)

    def _browse_simulation_settings(self):
        selected = filedialog.askopenfilename(parent=self.window, title="HUDYAT settings or Recognition lab results", filetypes=[("JSON", "*.json")])
        if selected:
            self.simulation_settings.set(selected)

    def start_simulation(self):
        if not self.test_run.get().strip():
            messagebox.showinfo("Choose a model", "Choose a completed random-split run or exported model version first.", parent=self.window)
            return
        arguments = ["simulate", "--run", self.test_run.get(), "--hudyat", self.hudyat_path.get(), "--profile", self.simulation_profile.get()]
        if self.simulation_candidate.get().strip():
            arguments.extend(["--candidate", self.simulation_candidate.get().strip()])
        if self.simulation_settings.get().strip():
            arguments.extend(["--settings-file", self.simulation_settings.get().strip()])
        if self._target() is not None:
            arguments.extend(["--target", self._target()])
        self.simulation_url.set("")
        self._start(arguments, "simulation")

    def choose_test_model(self, path=None):
        if path is None:
            selected = filedialog.askdirectory(parent=self.window, initialdir=self.settings.path("runs_dir"), title="Choose a trained run or exported model version")
            if not selected:
                return
            path = Path(selected)
        self.test_run.set(str(path))
        self._load_test_classes()
        self.tabs.select(self.testing_tab)

    def _load_test_classes(self):
        path = Path(self.test_run.get())
        try:
            classes = read_json(path / "classes.json") if (path / "classes.json").exists() else read_json(path / "model.json")["classes"]
        except (OSError, ValueError, KeyError):
            classes = []
        self.target_box.configure(values=["(unscored)", *classes])
        if self.live_target.get() not in classes:
            self.live_target.set("(unscored)")

    def _target(self):
        return None if self.live_target.get() == "(unscored)" else self.live_target.get()

    def automated_test(self):
        if not self.test_run.get().strip():
            messagebox.showinfo("Choose a run", "Choose a completed training run first.", parent=self.window)
            return
        self._start(["test", "--run", self.test_run.get()], "testing")

    def start_live(self):
        if not self.test_run.get().strip():
            messagebox.showinfo("Choose a model", "Choose a completed run or exported model version first.", parent=self.window)
            return
        try:
            camera = int(self.live_camera.get())
            if camera < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Camera index", "Use a nonnegative camera index, usually 0.", parent=self.window)
            return
        self._load_test_classes()
        arguments = ["live-worker", "--run", self.test_run.get(), "--camera", str(camera), "--mode", self.live_mode.get(), "--control-stdin"]
        if self._target() is not None:
            arguments.extend(["--target", self._target()])
        self.live_ready = False
        with self.live_frame_lock:
            self.live_frame = None
        self.live_prediction.set("Loading model...")
        self._start(arguments, "live")

    def _send_live(self, command):
        if self.process is not None and self.job_kind == "live" and self.process.stdin:
            try:
                self.process.stdin.write(json.dumps(command) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                self.stop()

    def _live_mode_changed(self):
        self._send_live({"type": "mode", "mode": self.live_mode.get()})
        self._capture_controls()

    def _capture_controls(self):
        manual = self.live_ready and self.live_mode.get() == "manual" and self.job_kind == "live" and self.process is not None
        self.capture_button.configure(state="normal" if manual else "disabled")
        self.finish_button.configure(state="normal" if manual else "disabled")

    def _capture_key(self, event, action):
        if self.window.focus_get() and self.window.focus_get().winfo_class() in ("Entry", "TEntry", "Text", "TCombobox"):
            return
        if self.tabs.select() == str(self.testing_tab) and self.test_modes.index("current") == 1 and self.live_ready and self.live_mode.get() == "manual":
            self._send_live({"type": action})
            return "break"

    def _live_event(self, event):
        kind = event.get("type")
        if kind == "frame":
            if event.get("png"):
                try:
                    self.live_image = tk.PhotoImage(master=self.window, data=event["png"], format="png")
                    self.camera_preview.configure(image=self.live_image, text="")
                except tk.TclError:
                    self.camera_preview.configure(image="", text="Preview unavailable")
            rates = []
            if event.get("preview_fps") is not None:
                rates.append(f"Preview: {event['preview_fps']:.1f} FPS")
            if event.get("fps") is not None:
                rates.append(f"Detection: {event['fps']:.1f} FPS")
            rates.append(f"Left: {'yes' if event['left'] else 'no'} | Right: {'yes' if event['right'] else 'no'} | Detection: {event['detector_ms']:.1f} ms")
            self.live_diagnostics.set(" | ".join(rates) + (" | Capturing" if event["capturing"] else ""))
        elif kind == "ready":
            self.live_ready = True
            self.target_box.configure(values=["(unscored)", *event["classes"]])
            self.live_stop.configure(state="normal")
            self._capture_controls()
            self.live_feedback.set(f"{event['category']} / {event['variant']} ready. Lower hands between automatic attempts.")
        elif kind in ("attempt", "estimate"):
            if kind == "estimate" and monotonic() - self.last_scored_at < 1.5:
                return
            if kind == "attempt":
                self.last_scored_at = monotonic()
            row = event["row"] if kind == "attempt" else event
            predicted, confidence = row.get("predicted"), row.get("confidence")
            self.live_prediction.set(f"{predicted}  {confidence:.1%}" if predicted is not None else "No usable prediction")
            labels = {"correct": "Correct", "incorrect": "Incorrect", "rejected": "Below confidence threshold", "invalid": "Invalid capture", "unscored": "Unscored prediction"}
            self.live_feedback.set((labels.get(row.get("outcome"), "") + (": " + row["error"] if row.get("error") else "")) if kind == "attempt" else "Live estimate — not a scored attempt")
            self.feedback_label.configure(foreground="#197347" if row.get("outcome") == "correct" else "#a62d2d" if row.get("outcome") in ("incorrect", "invalid") else "#795a13")
            self.top_predictions.delete(*self.top_predictions.get_children())
            for guess in row.get("top3", []):
                self.top_predictions.insert("", "end", values=(guess["label"], f"{guess['probability'] * 100:.1f}"))
        elif kind == "status":
            self.live_feedback.set(event["message"])
        elif kind == "session_complete":
            self._append("Test session saved: " + event["path"])

    def _settings_page(self, frame):
        self._heading(frame, "Local tools", "Normally, extraction and training use the Python environment that launched this app. Optional interpreter paths let you use separately installed environments. Saved choices apply only to this checkout.")
        self._path_row(frame, "Extraction Python", self.extract_python, lambda: self.browse_python(self.extract_python))
        self._path_row(frame, "Train & test Python", self.train_python, lambda: self.browse_python(self.train_python))
        self._path_row(frame, "HUDYAT project", self.hudyat_path, lambda: self.browse_dir(self.hudyat_path, self.root.parent))
        self._path_row(frame, "TF.js package (optional)", self.tfjs_path, lambda: self.browse_dir(self.tfjs_path, self.root.parent))
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=10)
        ttk.Button(row, text="Save local settings", command=self.save_settings).pack(side="left")
        self._button(row, "Check extraction dependencies", lambda: self._start(["doctor"], "extraction")).pack(side="left", padx=8)
        self._button(row, "Check training dependencies", lambda: self._start(["doctor"], "training")).pack(side="left")
        ttk.Label(frame, text="Vocabulary and study paths: signs.csv and project.json. Train > Left/right mirroring defaults to All signs; Approved signs only follows mirror_ok in signs.csv. Changing the Train control needs no re-extraction. Editing either file requires extracting again.", wraplength=1000).pack(anchor="w", pady=12)
        ttk.Button(frame, text="Open project folder", command=lambda: self.open_path(self.root)).pack(anchor="w")

    def _variant_changed(self, event=None):
        self.export_path.set(str(self.settings.path("exports_dir") / self.variant.get()))
        self.coverage_status.set("Version changed. Scan again to refresh coverage.")

    def _append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _python(self, kind):
        configured = self.train_python.get().strip() if kind in ("training", "export", "testing", "live", "reporting", "simulation") else self.extract_python.get().strip() if kind in ("extraction", "verify") else ""
        executable = Path(configured or sys.executable)
        if executable.name.lower() == "pythonw.exe" and executable.with_name("python.exe").exists():
            executable = executable.with_name("python.exe")
        return str(executable)

    def _start(self, arguments, kind):
        if self.process is not None:
            return
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.cancel_file = self.root / "tmp" / "jobs" / (job_id + ".cancel")
        self.cancel_file.parent.mkdir(parents=True, exist_ok=True)
        command = [self._python(kind), "-u", "-m", "hudyat_train", "--root", str(self.root), "--cancel-file", str(self.cancel_file), *arguments]
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE if kind == "live" else subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE if kind == "live" else subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as error:
            messagebox.showerror("Could not start job", str(error), parent=self.window)
            return
        self.job_kind = kind
        self.status.set("Running: " + " ".join(arguments))
        self._append("\n> " + " ".join(arguments))
        self.progress.start(12)
        self.stop_button.configure(state="normal")
        for button in self.actions:
            button.configure(state="disabled")
        self.category_box.configure(state="disabled")
        self.variant_box.configure(state="disabled")
        process = self.process
        log_path = self.root / "reports" / "jobs" / (job_id + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if kind == "live":
            def collect_errors():
                with log_path.with_suffix(".stderr.log").open("w", encoding="utf-8") as saved:
                    for line in process.stderr:
                        saved.write(line)
                        saved.flush()
                        self.events.put(("line", line))
            threading.Thread(target=collect_errors, daemon=True).start()

        def collect():
            try:
                with log_path.open("w", encoding="utf-8") as saved:
                    for line in process.stdout:
                        if kind == "live":
                            try:
                                event = json.loads(line)
                                if event.get("type") == "frame":
                                    with self.live_frame_lock:
                                        self.live_frame = event
                                    continue
                                self.events.put(("live_event", event))
                                if event.get("type") == "estimate":
                                    continue
                            except (ValueError, AttributeError):
                                if '"png"' in line:
                                    continue
                        saved.write(line)
                        saved.flush()
                        self.events.put(("line", line))
            finally:
                self.events.put(("done", process.wait()))

        threading.Thread(target=collect, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "line":
                    self._append(value)
                    if value.startswith("RUN_PATH="):
                        self.run_path.set(value.strip().split("=", 1)[1])
                        self.test_run.set(self.run_path.get())
                        self._load_test_classes()
                    elif value.startswith("SIMULATION_URL="):
                        self.simulation_url.set(value.strip().split("=", 1)[1])
                        self.status.set("HUDYAT simulator running. Allow the camera in the browser.")
                elif kind == "live_event":
                    self._live_event(value)
                elif kind == "done":
                    self.process = None
                    self.live_ready = False
                    with self.live_frame_lock:
                        self.live_frame = None
                    self.live_stop.configure(state="disabled")
                    self._capture_controls()
                    if self.job_kind == "live":
                        self.camera_preview.configure(image="", text="Camera stopped. Results saved in reports/testing/.")
                        self.live_image = None
                    self.progress.stop()
                    self.stop_button.configure(state="disabled")
                    for button in self.actions:
                        button.configure(state="normal")
                    self.category_box.configure(state="readonly")
                    self.variant_box.configure(state="readonly")
                    self.status.set("Completed." if value == 0 else "Stopped." if value == 130 else "Job failed. See the log for details.")
                    if self.job_kind == "scan":
                        self._load_scan()
                    if self.job_kind in ("training", "testing", "live", "reporting", "simulation"):
                        self.results_view.refresh()
                    if self.closing:
                        self.window.destroy()
                        return
        except queue.Empty:
            pass
        if self.live_ready:
            with self.live_frame_lock:
                frame, self.live_frame = self.live_frame, None
            if frame is not None:
                self._live_event(frame)
        self.window.after(16 if self.live_ready else 100, self._poll)

    def scan(self):
        self._start(["scan", "--variant", self.variant.get()], "scan")

    def _load_scan(self):
        path = self.root / "reports" / f"scan-{self.variant.get()}.json"
        if not path.exists():
            return
        try:
            self.scan_report = read_json(path)
            self.coverage.delete(*self.coverage.get_children())
            for row in self.scan_report["coverage"]:
                self.coverage.insert("", "end", iid=row["category"], values=(row["name"], row["sources"],
                                     f"{row['classes_present']}/{row['classes_expected']}", ", ".join(row["signers"]) or "none"))
            self.coverage_status.set(f"{self.scan_report['sources']} eligible raw files | {len(self.scan_report['issues'])} errors | {len(self.scan_report['excluded'])} excluded files. Counts are before extraction quality checks.")
            if self.category.get() in self.coverage.get_children():
                self.coverage.selection_set(self.category.get())
        except (ValueError, KeyError) as error:
            self._append("Could not read scan report: " + str(error))

    def _coverage_selected(self, event=None):
        selection = self.coverage.selection()
        if not selection or not self.scan_report:
            return
        category = selection[0]
        self.category.set(category)
        row = next(row for row in self.scan_report["coverage"] if row["category"] == category)
        self.sign_counts.delete(*self.sign_counts.get_children())
        for sign in row["signs"]:
            self.sign_counts.insert("", "end", values=(sign["label"], *[sign["counts"][s] for s in self.settings.get("expected_signers")]))

    def extract(self):
        self._start(["extract", "--category", self.category.get(), "--variant", self.variant.get()], "extraction")

    def train(self):
        if self.category.get() == "all":
            messagebox.showinfo("Choose a category", "Train one category at a time.", parent=self.window)
            return
        try:
            epochs, seed = int(self.epochs.get()), int(self.seed.get())
            copies = int(self.augmentation_copies.get()) if self.augmentation_enabled.get() else 0
            if not 1 <= epochs <= 150 or copies < 0 or (self.augmentation_enabled.get() and copies == 0):
                raise ValueError()
        except ValueError:
            messagebox.showerror("Training settings", "Use 1-150 epochs, an integer seed, and a positive whole number of augmented copies (or disable augmentation).", parent=self.window)
            return
        if copies != 5 and not self.development.get():
            messagebox.showerror("Study augmentation", "Study runs use 5 augmented copies per original. Enable Development run to experiment with a different count or disable augmentation.", parent=self.window)
            return
        arguments = ["train", "--category", self.category.get(), "--variant", self.variant.get(), "--protocol", self.protocol.get(),
                     "--epochs", str(epochs), "--seed", str(seed), "--mirroring", MIRRORING_POLICIES[self.mirroring.get()],
                     "--augmentation-copies", str(copies)]
        if self.development.get():
            arguments.append("--development")
        self._start(arguments, "training")

    def export(self):
        if not self.run_path.get().strip():
            messagebox.showinfo("Choose a run", "Choose a completed random-split run first.", parent=self.window)
            return
        arguments = ["export", "--run", self.run_path.get(), "--output", self.export_path.get()]
        if self.allow_development.get():
            arguments.append("--allow-development")
        self._start(arguments, "export")

    def verify(self):
        arguments = ["verify", "--hudyat", self.hudyat_path.get(), "--bundle", self.export_path.get()]
        if self.tfjs_path.get().strip():
            arguments.extend(["--tfjs", self.tfjs_path.get()])
        self._start(arguments, "verify")

    def stop(self):
        if self.process is not None and self.cancel_file:
            write_bytes(self.cancel_file, b"stop\n")
            self.status.set("Stopping at the next frame or training batch...")
            self.stop_button.configure(state="disabled")

    def close(self):
        if self.process is not None:
            self.closing = True
            self.stop()
        else:
            self.window.destroy()

    def browse_dir(self, variable, initial):
        result = filedialog.askdirectory(parent=self.window, initialdir=initial)
        if result:
            variable.set(result)

    def browse_python(self, variable):
        result = filedialog.askopenfilename(parent=self.window, title="Choose Python executable", filetypes=[("Python", "python*.exe"), ("All files", "*")])
        if result:
            variable.set(result)

    def save_settings(self):
        write_json(self.root / "settings.local.json", {"extraction_python": self.extract_python.get().strip(),
                   "training_python": self.train_python.get().strip(), "hudyat_path": self.hudyat_path.get().strip(), "tfjs_path": self.tfjs_path.get().strip()})
        self.status.set("Local tool settings saved.")

    def open_path(self, path):
        path = Path(path)
        if not path.exists():
            messagebox.showinfo("Folder not available", f"This folder has not been created yet:\n{path}", parent=self.window)
            return
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])


def create_window():
    return tk.Tk()


def main(root: Path = ROOT, testing_run=None, camera=0, mode="automatic"):
    window = create_window()
    try:
        app = App(window, root)
        if testing_run:
            app.choose_test_model(testing_run)
            app.live_camera.set(str(camera))
            app.live_mode.set(mode)
            app.test_modes.select(1)
    except (OSError, ValueError, KeyError) as error:
        window.withdraw()
        messagebox.showerror("HUDYAT Train", str(error), parent=window)
        window.destroy()
        return
    window.mainloop()
