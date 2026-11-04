from hudyat_train.reporting import result_report
from hudyat_train.testing import summarize_attempts


def test_reports_contain_graphs_csv_tables_and_escaped_html(tmp_path):
    rows = [{"attempt": 1, "expected": "A", "predicted": "A", "confidence": 0.95, "accepted": True,
             "outcome": "correct", "model_ms": 3.0, "pipeline_ms": 4.0, "source": "<private & local>"},
            {"attempt": 2, "expected": "B", "predicted": "A", "confidence": 0.7, "accepted": False,
             "outcome": "rejected", "model_ms": 5.0, "pipeline_ms": 7.0, "source": "clip.mp4"}]
    metrics = summarize_attempts(rows, ["A", "B"])
    result_report(tmp_path, {"category": "alphabet", "variant": "sequence", "kind": "automated_test", "status": "complete", "development": True}, metrics, rows)
    for file in ["summary.csv", "per-class.csv", "per-class-metrics.png", "confusion-matrix.png", "confidence-latency.png", "report.html"]:
        assert (tmp_path / file).stat().st_size > 0
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Summary table" in html and "Per-class table" in html and "Attempt table" in html
    assert "&lt;private &amp; local&gt;" in html and "<private & local>" not in html
