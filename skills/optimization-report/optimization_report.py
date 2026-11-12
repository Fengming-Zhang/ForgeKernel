#!/usr/bin/env python3
"""Append-only optimization log for one ForgeKernel run.

The log has one entry per optimization point, in the order the points were recorded.
Every entry carries the same three fields:

    Observation          what was actually measured or seen
    Decision & action    the decision id, the technique, what changed
    Result               the verdict, the measured delta, correctness

Two files sit side by side, ``<op>.jsonl`` (the machine-readable, append-only source
of truth) and ``<op>.html`` (the rendered document a human reads).  The harness also
calls :func:`sync_for` after every trajectory record it appends, so an entry exists
even when an agent turn was cut short.

Standalone on purpose: an agent runs this from the run machine's checkout, the GPU
job never needs it, and it imports only the standard library.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from pathlib import Path

ROUTES = ("student", "teacher")
STUDENT = "student"
TEACHER = "teacher"
# Fields a sync may overwrite: the objective outcome the harness measured.  The
# agent's own observation/action text is never overwritten by a sync.
SYNC_OVERWRITE = {"result", "verdict", "metric_before", "metric_after", "evidence"}
AGENT_OVERWRITE = SYNC_OVERWRITE | {"observation", "action", "decision_id", "ts"}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def data_path_for(html_path: str | Path) -> Path:
    return Path(html_path).with_suffix(".jsonl")


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _write_jsonl(path: str | Path, entries: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def _normalise(entry: dict) -> dict:
    iter_value = entry.get("iter")
    if isinstance(iter_value, str) and iter_value.strip().isdigit():
        iter_value = int(iter_value.strip())
    return {
        "ts": entry.get("ts") or _now(),
        "route": entry.get("route") if entry.get("route") in ROUTES else STUDENT,
        "iter": iter_value,
        "decision_id": entry.get("decision_id") or "",
        "observation": entry.get("observation") or "",
        "action": entry.get("action") or "",
        "result": entry.get("result") or "",
        "verdict": entry.get("verdict") or "",
        "metric_before": entry.get("metric_before"),
        "metric_after": entry.get("metric_after"),
        "evidence": entry.get("evidence") or "",
    }


def _key(entry: dict) -> tuple[str, str]:
    """One optimization point = (route, iteration).  ``iter`` is the inner iteration
    for a student row and the meta step for a teacher row."""
    return (entry.get("route") or STUDENT, str(entry.get("iter")))


def _merge(old: dict, new: dict, overwrite: set[str]) -> dict:
    out = dict(old)
    for key, value in new.items():
        if key in overwrite:
            # A sync may not have a metric for a row (a correctness FAIL, a scaffold
            # step); keep what the entry already states rather than erasing it.
            if value is None and key in ("metric_before", "metric_after"):
                continue
            out[key] = value
        elif value not in (None, "") and out.get(key) in (None, ""):
            out[key] = value
    return out


def upsert(entries: list[dict], incoming: dict,
           overwrite: set[str]) -> tuple[list[dict], bool, int]:
    """Insert or merge one entry; returns (entries, was_new, index)."""
    incoming = _normalise(incoming)
    for i, old in enumerate(entries):
        if _key(old) == _key(incoming):
            entries[i] = _merge(old, incoming, overwrite)
            return entries, False, i
    entries.append(incoming)
    return entries, True, len(entries) - 1


def append_entry(data_path: str | Path, entry: dict, *, overwrite: set[str] | None = None) -> dict:
    entries = _read_jsonl(data_path)
    entries, _, index = upsert(entries, entry, overwrite or AGENT_OVERWRITE)
    _write_jsonl(data_path, entries)
    return entries[index]


def _fmt_ms(value) -> str:
    try:
        return f"{float(value):.4f} ms"
    except (TypeError, ValueError):
        return ""


def _student_entry(rec: dict) -> dict:
    it = rec.get("iter")
    verdict = rec.get("verdict") or ""
    before = rec.get("metric_before")
    after = rec.get("device_median_ms")
    if after is None:
        after = rec.get("bwd_median_ms")
    bits = []
    if before is not None and after is not None:
        bits.append(f"latency {_fmt_ms(before)} -> {_fmt_ms(after)}")
    elif after is not None:
        bits.append(f"latency {_fmt_ms(after)}")
    ratio = rec.get("vs_ref_device")
    ratio_name = "device"
    if ratio is None:
        ratio = rec.get("vs_ref")
        ratio_name = "e2e"
    if ratio is not None:
        bits.append(f"{float(ratio):.4f}x the reference ({ratio_name})")
    correctness = rec.get("correctness")
    if correctness is True:
        bits.append("correctness PASS")
    elif correctness is False:
        bits.append("correctness FAIL")
    if rec.get("error"):
        bits.append(f"failure: {rec['error']}")
    prof = rec.get("profiler") or {}
    if isinstance(prof, dict) and prof:
        lead = []
        for key, value in prof.items():
            name = str(key).lower()
            if any(tag in name for tag in ("tensor", "dram", "sm__", "stall", "occupancy")):
                lead.append(f"{key}={value}")
            if len(lead) >= 4:
                break
        if lead:
            bits.append("NCU: " + ", ".join(lead))
    observation = "; ".join(bits) or f"iteration {it}: no measurement recorded"
    action = rec.get("change") or "(change not recorded)"
    if verdict == "NEUTRAL":
        result = "NEUTRAL (committed, not evaluated — build phase)"
    else:
        result = verdict or "recorded"
        if after is not None:
            result += f", {_fmt_ms(after)}"
        if correctness is True:
            result += ", correctness PASS"
        elif correctness is False:
            result += ", correctness FAIL"
        if rec.get("error") and verdict != "NEUTRAL":
            result += f", {rec['error']}"
    return {
        "ts": _now(),
        "route": STUDENT,
        "iter": it,
        "decision_id": rec.get("decision_id") or "",
        "observation": observation,
        "action": action,
        "result": result,
        "verdict": verdict,
        "metric_before": before,
        "metric_after": after,
        "evidence": f"status.jsonl iter {it}",
    }


def _teacher_entry(rec: dict) -> dict:
    step = rec.get("meta_step")
    bits = []
    if rec.get("cause_class"):
        bits.append(str(rec["cause_class"]))
    if rec.get("deficient_locus"):
        bits.append(f"@ {rec['deficient_locus']}")
    if rec.get("best_before") is not None:
        bits.append(f"best before {_fmt_ms(rec['best_before'])}")
    observation = " ".join(bits) or f"meta step {step}: no measurement recorded"
    action_bits = []
    if rec.get("patch_class"):
        action_bits.append(str(rec["patch_class"]))
    if rec.get("changed_entries"):
        action_bits.append("changed " + ", ".join(map(str, rec["changed_entries"])))
    if rec.get("patch_direction"):
        action_bits.append(str(rec["patch_direction"]))
    action = ": ".join(action_bits[:1]) + ((" — " + "; ".join(action_bits[1:])) if action_bits[1:] else "")
    outcome = rec.get("outcome") or "PENDING"
    result = outcome
    if rec.get("best_after") is not None:
        result += f", best after {_fmt_ms(rec['best_after'])}"
    if outcome == "PENDING":
        result += " (awaiting the segment that tests it)"
    return {
        "ts": _now(),
        "route": TEACHER,
        "iter": step,
        "decision_id": f"meta{step}" if step is not None else "",
        "observation": observation,
        "action": action or "(patch not recorded)",
        "result": result,
        "verdict": outcome,
        "metric_before": rec.get("best_before"),
        "metric_after": rec.get("best_after"),
        "evidence": f"meta_status.jsonl step {step}",
    }


def _targets(path: str | Path) -> tuple[Path, Path, str, str] | None:
    """``(data, html, op, run_dir)`` for a trajectory file, or None when the file is
    not inside a run's operator tree (a unit test, a hand-made tree)."""
    p = Path(path)
    if p.name == "status.jsonl" and p.parent.name == "student":
        op_dir = p.parent.parent          # <run>/operators/<op>
        run_dir = op_dir.parent.parent    # <run>
    elif p.name == "meta_status.jsonl":
        op_dir = p.parent                 # <run>/operators/<op>
        run_dir = op_dir.parent.parent    # <run>
    else:
        return None
    if op_dir.parent.name != "operators":
        return None
    report_dir = run_dir / "report"
    op = op_dir.name
    return report_dir / f"{op}.jsonl", report_dir / f"{op}.html", op, str(run_dir)


def sync(data_path: str | Path, *, status_path: str | Path | None = None,
         meta_status_path: str | Path | None = None) -> int:
    """Merge the objective trajectory into the log; returns the number of rows added."""
    entries = _read_jsonl(data_path)
    added = 0
    if status_path is not None:
        for rec in _read_jsonl(status_path):
            entries, new, _ = upsert(entries, _student_entry(rec), SYNC_OVERWRITE)
            added += int(new)
    if meta_status_path is not None:
        for rec in _read_jsonl(meta_status_path):
            entries, new, _ = upsert(entries, _teacher_entry(rec), SYNC_OVERWRITE)
            added += int(new)
    _write_jsonl(data_path, entries)
    return added


def sync_for(trajectory_path: str | Path) -> bool:
    """Sync and re-render after one trajectory file changed.  Best-effort: a log
    problem must never take the loop down."""
    try:
        targets = _targets(trajectory_path)
        if targets is None:
            return False
        data, html_path, op, run_dir = targets
        student = Path(trajectory_path).name == "status.jsonl"
        sync(data,
             status_path=trajectory_path if student else None,
             meta_status_path=None if student else trajectory_path)
        render_html(data, html_path, op=op, run_dir=run_dir)
        return True
    except Exception:  # noqa: BLE001 — bookkeeping, never a reason to stop a run
        return False


# -- rendering ---------------------------------------------------------------------


def _esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _badge(entry: dict) -> tuple[str, str]:
    verdict = (entry.get("verdict") or "").upper()
    route = entry.get("route") or STUDENT
    if route == TEACHER:
        if verdict in ("KEEP_PATCH", "KB_CERTIFIED", "AUDIT_CLEAN"):
            return "good", verdict or "teacher"
        if verdict in ("REVERT_PATCH", "KB_UNCERTIFIED", "UNPRICED_REOPEN"):
            return "bad", verdict or "teacher"
        return "neutral", verdict or "teacher"
    if verdict == "KEEP":
        return "good", "KEEP"
    if verdict == "REVERT":
        return "bad", "REVERT"
    if verdict == "NEUTRAL":
        return "neutral", "NEUTRAL"
    return "neutral", verdict or "recorded"


def _field(label: str, text: str, cls: str) -> str:
    return (f'<div class="field {cls}"><div class="label">{_esc(label)}</div>'
            f'<div class="text">{_esc(text)}</div></div>')


def _entry_html(index: int, entry: dict) -> str:
    tone, badge = _badge(entry)
    route = entry.get("route") or STUDENT
    iter_value = entry.get("iter")
    iter_label = f"iteration {iter_value}" if route == STUDENT else f"meta step {iter_value}"
    decision = entry.get("decision_id") or ""
    metrics = []
    if entry.get("metric_before") is not None:
        metrics.append(f"before {_fmt_ms(entry['metric_before'])}")
    if entry.get("metric_after") is not None:
        metrics.append(f"after {_fmt_ms(entry['metric_after'])}")
    metric_line = (" · ".join(metrics) + " · ") if metrics else ""
    return f"""
<article class="entry tone-{tone}" id="entry-{index}">
  <header class="entry-head">
    <span class="route">{_esc(route)}</span>
    <span class="iter">{_esc(iter_label)}</span>
    {f'<span class="decision">{_esc(decision)}</span>' if decision else ''}
    <span class="badge badge-{tone}">{_esc(badge)}</span>
    <span class="stamp">{_esc(entry.get('ts') or '')}</span>
  </header>
  <div class="metric-line">{_esc(metric_line + (entry.get('evidence') or ''))}</div>
  <div class="fields">
    {_field("Observation", entry.get("observation") or "(not recorded)", "observation")}
    {_field("Decision & action", entry.get("action") or "(not recorded)", "action")}
    {_field("Result", entry.get("result") or "(not recorded)", "result")}
  </div>
</article>"""


def render_html(data_path: str | Path, html_path: str | Path, *, op: str = "",
                run_dir: str = "", title: str = "") -> Path:
    entries = _read_jsonl(data_path)
    student = [e for e in entries if (e.get("route") or STUDENT) == STUDENT]
    teacher = [e for e in entries if e.get("route") == TEACHER]
    keeps = sum(1 for e in student if (e.get("verdict") or "").upper() == "KEEP")
    reverts = sum(1 for e in student if (e.get("verdict") or "").upper() == "REVERT")
    best = None
    for entry in student:
        value = entry.get("metric_after")
        if (entry.get("verdict") or "").upper() == "KEEP" and value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            best = value if best is None or value < best else best
    name = title or (op or Path(html_path).stem)
    body = "".join(_entry_html(i, entry) for i, entry in enumerate(entries, start=1))
    if not body:
        body = '<p class="empty">No optimization points recorded yet.</p>'
    subtitle = f'<div class="subtitle">{_esc(run_dir)}</div>' if run_dir else ""
    generated = _now()
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(name)} — optimization log</title>
<style>
  :root {{
    --bg:#f6f7f9; --panel:#ffffff; --ink:#1c2024; --muted:#5c6670; --line:#dde2e7;
    --good:#17663a; --good-bg:#e7f5ec; --bad:#9b1c1c; --bad-bg:#fbe9e9;
    --neutral:#5b4a12; --neutral-bg:#fbf3d9; --accent:#123a63;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#14171a; --panel:#1c2126; --ink:#e8ebee; --muted:#a2abb3; --line:#303840;
      --good:#8fd9a8; --good-bg:#1c3325; --bad:#f0a3a3; --bad-bg:#3a1f1f;
      --neutral:#e6cd85; --neutral-bg:#332c18; --accent:#9ec5ef;
    }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  main {{ max-width:1040px; margin:0 auto; padding:32px 20px 80px; }}
  h1 {{ margin:0 0 4px; font-size:28px; letter-spacing:-0.01em; }}
  .subtitle {{ color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
               font-size:12px; overflow-wrap:anywhere; }}
  .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
              gap:10px; margin:22px 0 28px; }}
  .tile {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  .tile .k {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
  .tile .v {{ font-size:22px; font-weight:650; margin-top:2px; }}
  .entry {{ background:var(--panel); border:1px solid var(--line); border-left-width:4px;
            border-radius:10px; margin:0 0 14px; overflow:hidden; }}
  .entry.tone-good {{ border-left-color:var(--good); }}
  .entry.tone-bad {{ border-left-color:var(--bad); }}
  .entry.tone-neutral {{ border-left-color:var(--neutral); }}
  .entry-head {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px;
                 padding:12px 16px; border-bottom:1px solid var(--line); }}
  .route {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }}
  .iter {{ font-weight:650; }}
  .decision {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px;
               color:var(--accent); }}
  .badge {{ font-size:11px; font-weight:700; padding:2px 8px; border-radius:999px; }}
  .badge-good {{ background:var(--good-bg); color:var(--good); }}
  .badge-bad {{ background:var(--bad-bg); color:var(--bad); }}
  .badge-neutral {{ background:var(--neutral-bg); color:var(--neutral); }}
  .stamp {{ margin-left:auto; color:var(--muted); font-size:11px; }}
  .metric-line {{ padding:8px 16px 0; color:var(--muted); font-size:12px;
                  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .fields {{ padding:10px 16px 16px; display:grid; gap:10px; }}
  .field {{ display:grid; grid-template-columns:170px 1fr; gap:14px; align-items:start; }}
  .label {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.07em;
            color:var(--muted); padding-top:2px; }}
  .text {{ white-space:pre-wrap; overflow-wrap:anywhere; }}
  .result .text {{ font-weight:600; }}
  .empty {{ color:var(--muted); }}
  footer {{ margin-top:32px; color:var(--muted); font-size:12px; }}
</style>
</head>
<body>
<main>
  <h1>{_esc(name)} — optimization log</h1>
  {subtitle}
  <section class="summary">
    <div class="tile"><div class="k">Optimization points</div><div class="v">{len(entries)}</div></div>
    <div class="tile"><div class="k">Student KEEP</div><div class="v">{keeps}</div></div>
    <div class="tile"><div class="k">Student REVERT</div><div class="v">{reverts}</div></div>
    <div class="tile"><div class="k">Teacher steps</div><div class="v">{len(teacher)}</div></div>
    <div class="tile"><div class="k">Best</div><div class="v">{_fmt_ms(best) if best is not None else '—'}</div></div>
  </section>
  <section class="timeline">
    {body}
  </section>
  <footer>Generated {_esc(generated)} by ForgeKernel's optimization log. One entry per
  optimization point: Observation → Decision &amp; action → Result.</footer>
</main>
</body>
</html>
"""
    out = Path(html_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(doc, encoding="utf-8")
    os.replace(tmp, out)
    return out


# -- CLI ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="optimization_report",
        description="Append one optimization point to a run's HTML log, or re-render it.")
    p.add_argument("--file", required=True, help="the HTML path, e.g. <run>/report/<op>.html")
    p.add_argument("--data", default=None, help="the JSONL sidecar (default: sibling .jsonl)")
    p.add_argument("--op", default="", help="operator/task name (default: the HTML stem)")
    p.add_argument("--run-dir", default="", help="the run directory, shown in the header")
    p.add_argument("--route", choices=ROUTES, default=STUDENT)
    p.add_argument("--iter", default=None, help="inner iteration, or meta step for teacher rows")
    p.add_argument("--decision", default="", help="decision id, e.g. D2.3 or D2.3+D2.4")
    p.add_argument("--observation", default="", help="what was measured or seen")
    p.add_argument("--action", default="", help="the decision and what changed")
    p.add_argument("--result", default="", help="verdict, delta, correctness")
    p.add_argument("--verdict", default="", help="KEEP | REVERT | NEUTRAL (optional)")
    p.add_argument("--metric-before", type=float, default=None)
    p.add_argument("--metric-after", type=float, default=None)
    p.add_argument("--evidence", default="", help="where the observation came from")
    p.add_argument("--render", action="store_true", help="only re-render the HTML")
    p.add_argument("--sync", action="store_true",
                   help="merge status.jsonl / meta_status.jsonl and re-render")
    p.add_argument("--status", default=None, help="path to status.jsonl (with --sync)")
    p.add_argument("--meta-status", default=None, help="path to meta_status.jsonl (with --sync)")
    args = p.parse_args(argv)

    html_path = Path(args.file).expanduser()
    data = Path(args.data).expanduser() if args.data else data_path_for(html_path)
    op = args.op or html_path.stem
    did = False

    if args.sync:
        added = sync(data, status_path=args.status, meta_status_path=args.meta_status)
        print(f"[optimization-report] synced {added} new row(s) into {data}", file=sys.stderr)
        did = True
    elif not args.render:
        if not (args.observation or args.action or args.result):
            p.error("append needs at least one of --observation / --action / --result")
        entry = {
            "route": args.route,
            "iter": args.iter,
            "decision_id": args.decision,
            "observation": args.observation,
            "action": args.action,
            "result": args.result or "pending (the harness records the measured result)",
            "verdict": args.verdict,
            "metric_before": args.metric_before,
            "metric_after": args.metric_after,
            "evidence": args.evidence,
        }
        saved = append_entry(data, entry)
        print(f"[optimization-report] recorded {saved['route']} {saved['iter']} "
              f"in {data}", file=sys.stderr)
        did = True

    if did or args.render:
        out = render_html(data, html_path, op=op, run_dir=args.run_dir)
        print(f"[optimization-report] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
