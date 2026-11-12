# optimization-report

Maintain the run's **optimization log**: one HTML document per operator task, one
entry per optimization point, in the order the points were recorded.

| | |
|---|---|
| file | `<run>/report/<op>.html` (the run directory, not the kernel workspace) |
| source | `<run>/report/<op>.jsonl` — append-only, one JSON object per entry |
| entries | `Observation` → `Decision & action` → `Result` |

The harness re-renders the HTML after every trajectory record, so an evaluated round
is present even when an agent turn was cut short. The agent's job is to add the part
the trajectory does not carry: **what it observed** and **why it chose the action**.

## When to write an entry

- **Student, every iteration.** At the end of the turn that implements a decision,
  append the entry with the observation that motivated it, the decision/technique,
  and `--result "pending"` (the driver measures it). At the start of the next turn,
  read `STATUS_JSONL` and update the *previous* entry with the measured result if you
  want to state it yourself; the harness fills it in either way.
- **Teacher, every step.** Reconnaissance findings, a distillation, a diagnosis, a
  patch, an M0 audit — each is an optimization point. Record the evidence you saw and
  the decision you made; the next segment's outcome is filled in when it is measured.
- Record failures too. A REVERT is an optimization path and often the most useful one.

## Append

```bash
python3 <repo>/skills/optimization-report/optimization_report.py \
  --file <run>/report/<op>.html --op <op> --run-dir <run> \
  --route student --iter <n> --decision <D id> \
  --observation "what was measured (numbers, NCU, correctness)" \
  --action "decision id + technique + what changed" \
  --result "KEEP/REVERT/NEUTRAL, delta, correctness"
```

`--route teacher --iter <meta step>` for a teacher step. Optional:
`--verdict`, `--metric-before`, `--metric-after`, `--evidence`.

Re-render only: `--render`. Merge the trajectory and re-render: `--sync --status
<run>/operators/<op>/student/status.jsonl --meta-status
<run>/operators/<op>/meta_status.jsonl`.

The script is standard-library only and can be run from any working directory. If it
is not available, write the same three fields into the HTML directly; keep the file
valid and append rather than rewriting history.

## Rules

- One entry per optimization point; never delete an entry. Correct an earlier claim by
  adding a new entry, not by editing history.
- Objective observations: numbers, verdicts, error signatures, NCU metrics. Put the
  reasoning in `Decision & action`, not in `Observation`.
- Never copy the reference's source, the milestone body, or any forbidden material
  into the log. Describe the decision in your own words.
- The log lives in the run directory, outside the kernel workspace: do not commit it,
  do not reference it from the kernel file.
