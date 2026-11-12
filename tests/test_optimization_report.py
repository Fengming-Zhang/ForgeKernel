"""The optimization log (skills/optimization-report): the run's HTML record of
every optimization point, and the standing rule that points agents at it."""

from __future__ import annotations

import json

from forgekernel.skills import optimization_report as report_log


def _write_status(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _student(iter_no, verdict, *, change="tile 128x256", ms=0.30, error=""):
    return {"iter": iter_no, "decision_id": f"D0.{iter_no}", "change": change,
            "correctness": verdict == "KEEP", "verdict": verdict,
            "device_median_ms": ms if verdict == "KEEP" else None,
            "metric_before": 0.33, "error": error, "profiler": {}}


def test_append_renders_the_three_fields_and_escapes_text(tmp_path):
    html = tmp_path / "report" / "demo.html"
    entry = {"route": "student", "iter": 1, "decision_id": "D0.1",
             "observation": "0.33 ms, tensor 71% <script>alert(1)</script>",
             "action": "pipeline stages 3 -> 4",
             "result": "pending"}
    saved = report_log.append_entry(report_log.data_path_for(html), entry)
    assert saved["iter"] == 1
    out = report_log.render_html(report_log.data_path_for(html), html, op="demo").read_text()
    for label in ("Observation", "Decision &amp; action", "Result"):
        assert label in out
    assert "&lt;script&gt;" in out and "<script>alert" not in out
    assert "pipeline stages 3 -&gt; 4" in out


def test_sync_for_creates_the_report_next_to_the_run_and_is_idempotent(tmp_path):
    run = tmp_path / "run"
    status = run / "operators" / "demo" / "student" / "status.jsonl"
    _write_status(status, [_student(1, "KEEP"), _student(2, "REVERT", error="max_diff=1e-2")])

    assert report_log.sync_for(status) is True
    html = run / "report" / "demo.html"
    assert html.is_file()
    text = html.read_text()
    assert "iteration 1" in text and "iteration 2" in text
    assert "KEEP" in text and "REVERT" in text and "max_diff=1e-2" in text

    # a second sync must not duplicate rows
    assert report_log.sync_for(status) is True
    rows = report_log._read_jsonl(run / "report" / "demo.jsonl")
    assert len(rows) == 2


def test_sync_keeps_the_agents_observation_and_fills_the_result(tmp_path):
    run = tmp_path / "run"
    status = run / "operators" / "demo" / "student" / "status.jsonl"
    _write_status(status, [_student(1, "KEEP")])
    data = run / "report" / "demo.jsonl"
    data.parent.mkdir(parents=True)
    report_log.append_entry(data, {
        "route": "student", "iter": 1, "decision_id": "D0.1",
        "observation": "the previous round plateaued at 0.33 ms",
        "action": "raise the pipeline depth", "result": "pending"})

    report_log.sync(data, status_path=status)
    report_log.sync(data, status_path=status)
    rows = report_log._read_jsonl(data)
    assert len(rows) == 1
    assert rows[0]["observation"] == "the previous round plateaued at 0.33 ms"
    assert rows[0]["action"] == "raise the pipeline depth"
    assert rows[0]["result"].startswith("KEEP")


def test_meta_status_rows_become_teacher_entries(tmp_path):
    run = tmp_path / "run"
    meta = run / "operators" / "demo" / "meta_status.jsonl"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({
        "meta_step": 2, "cause_class": "R2", "deficient_locus": "D0.3",
        "patch_class": "additive", "changed_entries": ["D0.3"],
        "patch_direction": "add TMA multicast", "outcome": "KEEP_PATCH",
        "best_before": 0.32, "best_after": 0.30}) + "\n", encoding="utf-8")

    assert report_log.sync_for(meta) is True
    text = (run / "report" / "demo.html").read_text()
    assert "meta step 2" in text
    assert "add TMA multicast" in text and "KEEP_PATCH" in text


def test_standing_rules_are_appended_and_rendered():
    from pathlib import Path
    from forgekernel.loop.driver import _standing_rules

    repo = Path(__file__).resolve().parents[1]
    text = _standing_rules(repo / "meta" / "student_prompt.md", {
        "{REPORT_PATH}": "/run/report/gemm.html",
        "{REPORT_TOOL}": "/repo/skills/optimization-report/optimization_report.py",
        "{OP}": "gemm",
    })
    assert "R1. Keep the task's optimization log" in text
    assert "/run/report/gemm.html" in text
    assert "{REPORT_PATH}" not in text and "{REPORT_TOOL}" not in text
