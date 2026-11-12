# Standing rules for every ForgeKernel agent turn

These rules apply to the student and to the teacher (reconnaissance, distillation,
diagnosis, patch, M0 audit) of every operator task. They are appended to each turn's
prompt after the role prompt and before the driver's iteration-specific notes.

## R1. Keep the task's optimization log

Every optimization point of this task must be recorded in the run's HTML log:

- file: `{REPORT_PATH}`
- one entry per optimization point, in three fields, in this order:
  1. **Observation** — what you actually measured or saw: the previous round's
     verdict, latency/ratio, NCU metric, correctness error, probe result. Objective
     numbers, not a theory.
  2. **Decision & action** — the decision id and the technique, what changed in the
     kernel or the milestone, and why it follows from the observation.
  3. **Result** — the verdict (`KEEP` / `REVERT` / `NEUTRAL`), the measured delta, and
     whether correctness passed. For a student iteration the result is measured after
     the turn ends; write `pending` and let the harness fill it, or record the previous
     round's result at the start of the next turn after reading `STATUS_JSONL`. For a
     teacher step, record the decision and leave the result to the segment that tests
     it.

Append with:

```bash
python3 {REPORT_TOOL} \
  --file {REPORT_PATH} --op {OP} \
  --route student --iter <n> --decision <D id> \
  --observation "<what you measured>" \
  --action "<decision + technique + what changed>" \
  --result "<KEEP/REVERT/NEUTRAL, delta, correctness | pending>"
```

Use `--route teacher --iter <meta step>` for a reconnaissance, distillation,
diagnosis or patch step. The full format and the fallback (write the HTML directly)
are in the skill `optimization-report`.

Rules for the log:

- **Never delete an entry and never rewrite history.** A wrong earlier claim is
  corrected by a later entry, not by editing the old one.
- **Record failures.** A REVERT is an optimization path, often the most useful one:
  say what was observed, what was tried, and what the measurement said.
- **Do not copy forbidden material into the log** (the reference's source, sealed
  artifacts, another branch). Describe the decision in your own words.
- The log lives in the run directory, outside the kernel workspace. Do not add it to
  the kernel file, do not commit it, and do not let it change the timed window.
