"""Browse experiment tables and graphs without importing numerical runtimes."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from .experiments import Registry


class ResultsView(ttk.Frame):
    def __init__(self, parent, settings, open_path, run_job, select_model):
        super().__init__(parent)
        self.settings, self.open_path, self.run_job, self.select_model = settings, open_path, run_job, select_model
        self.records_by_id, self.current, self.image = {}, None, None
        self.message = tk.StringVar(value="Training runs and test sessions are recorded automatically.")
        tools = ttk.Frame(self)
        tools.pack(fill="x", pady=(0, 8))
        ttk.Button(tools, text="Refresh tables", command=self.refresh).pack(side="left")
        ttk.Button(tools, text="Build older reports", command=lambda: run_job(["results", "--rebuild"], "reporting")).pack(side="left", padx=6)
        ttk.Button(tools, text="Open experiment CSV", command=lambda: open_path(settings.root / "reports" / "experiments.csv")).pack(side="left")
        ttk.Button(tools, text="Open result folder", command=lambda: open_path(Path(self.current["path"])) if self.current else None).pack(side="left", padx=6)
        ttk.Button(tools, text="Use model for testing", command=self.use_model).pack(side="right")
        columns = ("run", "kind", "category", "variant", "protocol", "status", "accuracy", "f1", "latency", "development")
        overview_frame = ttk.Frame(self)
        overview_frame.pack(fill="x")
        self.overview = ttk.Treeview(overview_frame, columns=columns, show="headings", height=5)
        for name, title, width in [("run", "Run / session", 175), ("kind", "Experiment", 120), ("category", "Category", 100), ("variant", "Version", 75),
                                   ("protocol", "Protocol", 105), ("status", "Status", 85), ("accuracy", "Raw top-1 %", 85),
                                   ("f1", "Macro F1 %", 85), ("latency", "Model ms", 80), ("development", "Development", 95)]:
            self.overview.heading(name, text=title)
            self.overview.column(name, width=width, minwidth=55, stretch=True)
        self.overview.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(overview_frame, command=self.overview.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(overview_frame, orient="horizontal", command=self.overview.xview)
        self.overview.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        horizontal.grid(row=1, column=0, sticky="ew")
        overview_frame.columnconfigure(0, weight=1)
        self.overview.bind("<<TreeviewSelect>>", self._selected)
        ttk.Label(self, textvariable=self.message, wraplength=1000).pack(anchor="w", pady=7)
        self.details = ttk.Notebook(self)
        self.details.pack(fill="both", expand=True)
        summary, per_class, attempts, graphs = [ttk.Frame(self.details, padding=5) for _ in range(4)]
        for title, frame in zip(("Summary table", "Per-class table", "Attempt table", "Graphs"), (summary, per_class, attempts, graphs)):
            self.details.add(frame, text=title)
        self.summary = self._tree(summary, [("metric", "Metric", 260), ("value", "Value", 600)])
        self.per_class = self._tree(per_class, [("label", "Sign", 200), ("support", "Test samples", 100),
                                              ("precision", "Precision %", 120), ("recall", "Recall %", 120), ("f1", "F1 %", 120)])
        self.attempts = self._tree(attempts, [("expected", "Expected", 130), ("predicted", "Predicted", 130), ("confidence", "Confidence %", 100),
                                            ("outcome", "Result", 95), ("model_ms", "Model ms", 90), ("source", "Source", 330)])
        graph_tools = ttk.Frame(graphs)
        graph_tools.pack(fill="x")
        self.graph_name = tk.StringVar()
        self.graph_choice = ttk.Combobox(graph_tools, textvariable=self.graph_name, state="readonly", width=32)
        self.graph_choice.pack(side="left")
        self.graph_choice.bind("<<ComboboxSelected>>", self.show_graph)
        ttk.Button(graph_tools, text="Open full-size graph", command=self.open_graph).pack(side="left", padx=6)
        ttk.Button(graph_tools, text="Open HTML report", command=self.open_report).pack(side="left")
        self.preview = ttk.Label(graphs, text="Select an experiment to view its graphs.", anchor="center")
        self.preview.pack(fill="both", expand=True, pady=5)
        self.after(500, self.refresh)

    def _tree(self, parent, columns):
        tree = ttk.Treeview(parent, columns=[c[0] for c in columns], show="headings", height=4)
        for name, title, width in columns:
            tree.heading(name, text=title)
            tree.column(name, width=width)
        scroll = ttk.Scrollbar(parent, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return tree

    def refresh(self):
        selected = self.overview.selection()
        records = Registry(self.settings.root).reconcile(self.settings.path("runs_dir"))
        self.records_by_id = {r["id"]: r for r in records}
        self.overview.delete(*self.overview.get_children())
        for r in records:
            m = r["metrics"]
            percent = lambda key: f"{m[key] * 100:.1f}" if isinstance(m.get(key), (int, float)) else "—"
            latency = m.get("latency", {}).get("mean_model_ms")
            self.overview.insert("", "end", iid=r["id"], values=(Path(r["path"]).name, r["kind"], r.get("category"), r.get("variant"), r.get("protocol"),
                                  r.get("status"), percent("accuracy"), percent("macro_f1"), f"{latency:.1f}" if latency is not None else "—", "Yes" if r.get("development") else "No"))
        if selected and selected[0] in self.records_by_id:
            self.overview.selection_set(selected[0])
        elif records:
            self.overview.selection_set(records[0]["id"])
        self.message.set(f"{len(records)} recorded experiments. Unscored or untested metrics display as —. Full comparison table: reports/experiments.csv")

    def _selected(self, event=None):
        selected = self.overview.selection()
        if not selected:
            return
        self.current = self.records_by_id[selected[0]]
        record, metrics = self.current, self.current["metrics"]
        directory = Path(record["path"])
        for tree in (self.summary, self.per_class, self.attempts):
            tree.delete(*tree.get_children())
        for key in ("path", "kind", "category", "variant", "protocol", "status", "started_at", "completed_at", "development", "dataset_id", "source_model"):
            if key in record:
                self.summary.insert("", "end", values=(key, record[key]))
        for key, value in metrics.items():
            if not isinstance(value, (dict, list)):
                self.summary.insert("", "end", values=(key, value if value is not None else "unscored"))
        for key, value in metrics.get("latency", {}).items():
            self.summary.insert("", "end", values=("latency." + key, value))
        for key, value in record.get("parameters", {}).items():
            self.summary.insert("", "end", values=("parameter." + key, value))
        for row in metrics.get("per_class", []):
            self.per_class.insert("", "end", values=(row["label"], row["support"], *[f"{row[k] * 100:.1f}" for k in ("precision", "recall", "f1")]))
        table = next((directory / name for name in ("attempts.csv", "predictions.csv") if (directory / name).exists()), None)
        if table:
            with table.open(newline="", encoding="utf-8-sig") as stream:
                for i, row in enumerate(csv.DictReader(stream)):
                    if i >= 500:
                        break
                    expected = row.get("expected", row.get("actual", ""))
                    confidence = row.get("confidence", "")
                    confidence = f"{float(confidence) * 100:.1f}" if confidence else "—"
                    self.attempts.insert("", "end", values=(expected, row.get("predicted", ""), confidence, row.get("outcome", ""), row.get("model_ms", ""), row.get("source", "")))
        graphs = sorted(p.name for p in directory.glob("*.png"))
        self.graph_choice.configure(values=graphs)
        self.graph_name.set(graphs[0] if graphs else "")
        self.show_graph()

    def show_graph(self, event=None):
        if not self.current or not self.graph_name.get():
            self.preview.configure(image="", text="Graphs are saved when the run or session finishes. Refresh tables to see current attempts.")
            self.image = None
            return
        try:
            image = tk.PhotoImage(master=self, file=str(Path(self.current["path"]) / self.graph_name.get()))
            scale = max(1, math.ceil(image.width() / 860), math.ceil(image.height() / 260))
            self.image = image.subsample(scale)
            self.preview.configure(image=self.image, text="")
        except tk.TclError:
            self.preview.configure(image="", text="Graph preview unavailable. Use Open full-size graph.")

    def open_graph(self):
        if self.current and self.graph_name.get():
            self.open_path(Path(self.current["path"]) / self.graph_name.get())

    def open_report(self):
        if self.current:
            self.open_path(Path(self.current["path"]) / "report.html")

    def use_model(self):
        if self.current:
            self.select_model(Path(self.current.get("source_model") or self.current["path"]))
