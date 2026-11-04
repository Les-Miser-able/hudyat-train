"""Portable PNG graphs, CSV tables, and an offline HTML report for each result."""
from __future__ import annotations

from html import escape
from pathlib import Path

from .experiments import write_csv
from .storage import write_bytes


def _table(rows, columns=None):
    if not rows:
        return "<p>No observations available.</p>"
    columns = columns or list(rows[0])
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{escape(str(c))}</th>" for c in columns) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>" for row in rows) + "</tbody></table></div>"


def result_report(directory: Path, state: dict, metrics: dict, attempts=None):
    """Additional report artifacts; never change the evaluated metrics/checkpoint files."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    per_class = metrics.get("per_class", [])
    write_csv(directory / "per-class.csv", per_class, ["label", "support", "precision", "recall", "f1"])
    summary_rows = [{"metric": key, "value": value} for key, value in metrics.items() if isinstance(value, (int, float, str)) or value is None]
    summary_rows += [{"metric": "latency." + key, "value": value} for key, value in metrics.get("latency", {}).items()]
    summary_rows += [{"metric": key, "value": state[key]} for key in
                     ("training_sources", "generated_training_samples", "mirrored_training_samples", "augmented_training_samples", "validation_sources", "test_sources", "epochs_completed", "dataset_id") if key in state]
    write_csv(directory / "summary.csv", summary_rows, ["metric", "value"])
    if per_class:
        fig, axis = plt.subplots(figsize=(max(8, len(per_class) * 0.45), 4.8))
        x = np.arange(len(per_class))
        for i, (name, title) in enumerate([( "precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
            axis.bar(x + (i - 1) * 0.25, [r[name] for r in per_class], width=0.25, label=title)
        axis.set(xticks=x, xticklabels=[r["label"] for r in per_class], ylim=(0, 1.05), ylabel="Score", title="Per-class results")
        axis.tick_params(axis="x", labelrotation=65)
        axis.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(directory / "per-class-metrics.png", dpi=160)
        plt.close(fig)
        matrix = np.asarray(metrics.get("confusion_matrix", []))
        if matrix.shape == (len(per_class), len(per_class)):
            fig, axis = plt.subplots(figsize=(max(6, len(per_class) * 0.35), max(5, len(per_class) * 0.35)))
            image = axis.imshow(matrix, cmap="Blues")
            axis.set(xticks=range(len(per_class)), yticks=range(len(per_class)), xticklabels=[r["label"] for r in per_class],
                     yticklabels=[r["label"] for r in per_class], xlabel="Predicted", ylabel="Expected", title="Raw top-1 predictions")
            axis.tick_params(axis="x", labelrotation=90)
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                              color="white" if matrix[i, j] > matrix.max() / 2 else "black")
            fig.colorbar(image, ax=axis)
            fig.tight_layout()
            fig.savefig(directory / "confusion-matrix.png", dpi=150)
            plt.close(fig)
    if attempts:
        valid = [r for r in attempts if r.get("predicted") is not None]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if valid:
            axes[0].plot(range(1, len(valid) + 1), [r["confidence"] for r in valid], marker=".", linewidth=1)
            threshold = metrics.get("confidence_threshold", 0.85)
            axes[0].axhline(threshold, color="#ad6b00", linestyle="--", label=f"Threshold {threshold:.0%}")
            axes[0].legend()
            axes[1].plot(range(1, len(valid) + 1), [r.get("model_ms", 0) for r in valid], marker=".", linewidth=1, label="Model only")
            pipeline = [(i + 1, r["pipeline_ms"]) for i, r in enumerate(valid) if r.get("pipeline_ms") is not None]
            if pipeline:
                axes[1].plot([p[0] for p in pipeline], [p[1] for p in pipeline], marker=".", linewidth=1, label="Input preparation + model")
                axes[1].legend()
        axes[0].set(xlabel="Prediction", ylabel="Top-1 confidence", ylim=(0, 1.05))
        axes[1].set(xlabel="Prediction", ylabel="Python inference time (ms)")
        if 0 < len(valid) <= 10:
            for axis in axes:
                axis.set_xticks(range(1, len(valid) + 1))
            if len(valid) == 1:
                for axis in axes:
                    axis.set_xlim(0.5, 1.5)
        elif not valid:
            for axis in axes:
                axis.text(0.5, 0.5, "No valid model predictions", ha="center", va="center", transform=axis.transAxes)
        fig.suptitle("Prediction confidence and Python inference timing")
        fig.tight_layout()
        fig.savefig(directory / "confidence-latency.png", dpi=160)
        plt.close(fig)
    title = f"{state.get('category', '')} / {state.get('variant', '')} / {state.get('kind', state.get('protocol', 'results'))}"
    graphs = [p for p in directory.glob("*.png") if p.name != "preview.png"]
    parameters = [{"parameter": key, "value": value} for key, value in state.get("parameters", {}).items()]
    if parameters:
        write_csv(directory / "parameters.csv", parameters)
    links = " · ".join(f"<a href='{escape(p.name, quote=True)}'>{escape(p.name)}</a>" for p in sorted(directory.glob("*.csv")))
    pages = ["<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
             f"<title>{escape(title)}</title><style>body{{font:15px system-ui;margin:32px;color:#202923}}h1{{font-size:24px}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ddd;text-align:left}}th{{background:#f1f1e9}}img{{max-width:100%;height:auto}}.scroll{{overflow:auto;max-height:480px}}section{{margin:24px 0}}</style>",
             f"<h1>{escape(title)}</h1><p>Status: {escape(state.get('status', 'complete'))}. Development: {escape(str(state.get('development', False)))}.</p>",
             f"<p>{escape(state.get('measurement_scope', 'Python desktop measurements are separate from HUDYAT browser response-time validation. Unlabelled predictions are excluded from accuracy; rejected and invalid attempts are reported separately.'))}</p>",
             f"<p>{links}</p><section><h2>Summary table</h2>{_table(summary_rows)}</section>",
             f"<section><h2>Per-class table</h2>{_table(per_class)}</section>"]
    if metrics.get("untested_classes"):
        pages.append("<p>Classes without ground-truth attempts: " + escape(", ".join(metrics["untested_classes"])) + ". This is partial coverage.</p>")
    if parameters:
        pages.append("<section><h2>Parameters</h2>" + _table(parameters) + "</section>")
    pages.extend(f"<section><h2>{escape(p.stem.replace('-', ' ').title())}</h2><img src='{escape(p.name, quote=True)}'></section>" for p in graphs)
    if attempts:
        pages.append("<section><h2>Attempt table</h2>" + _table(attempts, ["attempt", "expected", "predicted", "confidence", "outcome", "model_ms", "pipeline_ms", "source"]) + "</section>")
    write_bytes(directory / "report.html", "\n".join(pages).encode("utf-8"))


def loso_report(directory, summary, state):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np
    rows = [{"test_signer": r["test_signer"], "validation_signer": r["validation_signer"],
             **{key: r[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}, "samples": r["samples"]}
            for r in summary["folds"]]
    write_csv(directory / "loso-folds.csv", rows)
    write_csv(directory / "loso-summary.csv", [{"metric": k, **v} for k, v in summary["statistics"].items()])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(rows))
    axes[0].bar(x - 0.18, [r["accuracy"] for r in rows], width=0.36, label="Accuracy")
    axes[0].bar(x + 0.18, [r["macro_f1"] for r in rows], width=0.36, label="Macro F1")
    axes[0].set(xticks=x, xticklabels=[r["test_signer"] for r in rows], xlabel="Held-out signer", ylim=(0, 1.05))
    axes[0].legend()
    stats = summary["statistics"]
    axes[1].bar(range(len(stats)), [v["mean"] for v in stats.values()], yerr=[v["std"] for v in stats.values()], capsize=5)
    axes[1].set(xticks=range(len(stats)), xticklabels=[k.replace("macro_", "") for k in stats], ylim=(0, 1.1), ylabel="Fold mean ± sample SD")
    fig.tight_layout()
    fig.savefig(directory / "loso-metrics.png", dpi=160)
    plt.close(fig)
    result_report(directory, state, {**summary["pooled"], **{k: v["mean"] for k, v in stats.items()}})
