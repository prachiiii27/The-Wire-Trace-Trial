#!/usr/bin/env python3
"""
Group 6 - Stage 1 evaluation: summarise replay logs (time, contact events, success rate).

OFFLINE ONLY - no ROS, publishes nothing.  Reads the logs that replay_dls.py writes to
data/replays/ and prints one line per run plus the overall success rate.

For every run it reports
  direction   fwd (start -> end of wire) or rev (replay_dls.py --reverse)
  result      done / abort / not started (the last phase in the log)
  run_s       time spent in the "run" phase [s]  (same number replay_dls.py prints)
  e_mean/max  stick-tip tracking error against the taught path [mm] and max angle error [deg]
  touches     wire contacts the operator marked with SPACE/T when the buzzer sounded
              ("n/a" for logs recorded before touch marking existed)
  success     result == done AND touches == 0

Usage
  python3 stage1/evaluate_runs.py                       # all logs in data/replays/
  python3 stage1/evaluate_runs.py data/replays/replay_20261117_223942.csv [more logs]
  python3 stage1/evaluate_runs.py --forward-only        # only start -> end runs
The table is also saved to data/evaluation_summary.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

GROUP_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = GROUP_DIR / "data" / "replays"
OUT_FILE = GROUP_DIR / "data" / "evaluation_summary.csv"


def evaluate(path: Path) -> dict:
    df = pd.read_csv(path)
    run = df[df.phase == "run"]
    last = str(df.phase.iloc[-1]) if len(df) else "empty"
    row = {
        "log": path.name,
        "direction": "rev" if path.name.startswith("replay_rev") else "fwd",
        "result": last if len(run) else "not started",
        "run_s": None, "e_mean_mm": None, "e_max_mm": None, "e_max_deg": None,
        "touches": None, "success": False,
    }
    if len(run) == 0:
        return row
    row["run_s"] = round(float(run.t.iloc[-1] - run.t.iloc[0]), 1)
    row["e_mean_mm"] = round(float(run.e_mm.mean()), 2)
    row["e_max_mm"] = round(float(run.e_mm.max()), 2)
    row["e_max_deg"] = round(float(run.e_deg.max()), 2)
    if "touch" in df.columns:
        row["touches"] = int(df.touch.sum())
        row["success"] = bool(last == "done" and row["touches"] == 0)
    else:
        row["success"] = None      # no touch column: contact was not logged for this run
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="replay logs (default: all in data/replays/)")
    ap.add_argument("--forward-only", action="store_true", help="skip --reverse runs")
    args = ap.parse_args()

    paths = [Path(p) for p in args.logs] if args.logs else sorted(LOG_DIR.glob("replay*.csv"))
    if not paths:
        raise SystemExit(f"no replay logs found in {LOG_DIR}")
    rows = [evaluate(p) for p in paths]
    if args.forward_only:
        rows = [r for r in rows if r["direction"] == "fwd"]
    table = pd.DataFrame(rows)
    table["touches"] = table.touches.astype("Int64")
    shown = table.astype(object).where(table.notna(), "n/a")
    print(shown.to_string(index=False))

    judged = table[table.success.notna()]
    n_ok = int(judged.success.sum())
    print(f"\nruns with contact logging: {len(judged)}   successful (done, 0 touches): {n_ok}"
          f"   success rate: {100 * n_ok / max(len(judged), 1):.0f} %")
    ok = judged[judged.success == True]  # noqa: E712
    if len(ok):
        best = ok.loc[ok.run_s.idxmin()]
        print(f"fastest successful run: {best.log}  {best.run_s} s ({best.direction})")
    if not args.logs:
        table.to_csv(OUT_FILE, index=False)
        print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()
