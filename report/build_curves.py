#!/usr/bin/env python3
"""Build the training-curve report page from sealpose logs.

Pipeline:  logs/*_out.log  ->  report/curves.csv  (parse_logs.py)
                           ->  report/data.json
                           ->  report/curves.html (report/template.html + data)

Run from the repo root after any new job finishes:

    python report/build_curves.py

Add a run by passing --runs; the order given is the colour-slot order
(slot 1..8 of the categorical palette, assigned in fixed order).
Standard library only.
"""

import argparse
import collections
import csv
import datetime
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_RUNS = ["lin-base", "lin-seal", "gcn-base", "gcn-seal"]
SERIES_FIELDS = ["mpjpe", "p_mpjpe", "pck", "auc", "loss", "time_s",
                 "e_diff_ratio", "e_diff", "energy"]


def log_meta(logs_dir, run):
    """Params, wall-clock duration and start time, read from the sbatch header."""
    match = [f for f in os.listdir(logs_dir)
             if f.startswith(run + "_") and f.endswith("_out.log")]
    if not match:
        return {}
    txt = open(os.path.join(logs_dir, match[0]), encoding="utf-8",
               errors="replace").read()
    grab = lambda pat: (re.search(pat, txt).group(1) if re.search(pat, txt) else None)
    params = grab(r"Number of parameters: ([\d,]+)")
    p_loss = grab(r"Number of parameters \(loss-net\): ([\d,]+)")
    started, finished = grab(r"started\s*:\s*([\d\- :]+)"), grab(r"finished\s*:\s*([\d\- :]+)")
    duration = ""
    if started and finished:
        f = "%Y-%m-%d %H:%M:%S"
        d = (datetime.datetime.strptime(finished.strip(), f)
             - datetime.datetime.strptime(started.strip(), f))
        tot = int(d.total_seconds())
        duration = f"{tot//3600}h {(tot%3600)//60}m" if tot >= 3600 else f"{(tot%3600)//60}m {tot%60}s"
    n = lambda v: int(v.replace(",", "")) if v else None
    return dict(params=n(params), params_loss=n(p_loss),
                started=(started or "").strip(), duration=duration)


def short(n):
    if n is None:
        return ""
    return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.0f}K"


def build_data(csv_path, logs_dir, run_order):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["run"]].append(r)

    missing = [r for r in run_order if r not in by]
    if missing:
        sys.exit(f"runs not found in {csv_path}: {', '.join(missing)}")

    num = lambda v: float(v) if v not in ("", None) else None
    runs = []
    for slot, key in enumerate(run_order, start=1):
        rs = sorted(by[key], key=lambda x: int(x["epoch"]))
        ser = {f: [num(r[f]) for r in rs] for f in SERIES_FIELDS}
        mp = ser["mpjpe"]

        # Running best: the lowest MPJPE achieved up to and including each epoch.
        best_so_far, b = [], float("inf")
        for v in mp:
            b = min(b, v)
            best_so_far.append(round(b, 2))
        # A "spike" is an epoch that lands 1.5x worse than the best seen so far —
        # a size-free way to say how violently a run oscillates.
        spike = sum(1 for v, bb in zip(mp, best_so_far) if v > 1.5 * bb)
        bi = min(range(len(rs)), key=lambda i: mp[i])
        m = log_meta(logs_dir, key)

        runs.append(dict(
            key=key, slot=slot, job_id=rs[0]["job_id"],
            backbone=rs[0]["task_net"], type=rs[0]["type"], lr=rs[0]["lr"],
            energy_weight=rs[0]["energy_weight"] or None,
            lr_loss=rs[0]["lr_loss"] or None,
            params=m.get("params"), params_loss=m.get("params_loss"),
            params_short=short(m.get("params")),
            params_loss_short=("+" + short(m["params_loss"])) if m.get("params_loss") else "",
            duration=m.get("duration", ""),
            epochs=[int(r["epoch"]) for r in rs],
            saved=[int(r["saved"]) for r in rs],
            **ser,
            best_so_far=best_so_far, spike=spike, mpjpe_max=round(max(mp), 1),
            best=dict(epoch=int(rs[bi]["epoch"]), mpjpe=mp[bi],
                      p_mpjpe=ser["p_mpjpe"][bi], pck=ser["pck"][bi], auc=ser["auc"][bi]),
            last=dict(epoch=int(rs[-1]["epoch"]), mpjpe=mp[-1]),
            epoch_s=round(sum(x for x in ser["time_s"] if x) / len(rs), 1),
        ))
    return {"runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.join(ROOT, "logs"))
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                    help="run names in colour-slot order")
    ap.add_argument("--out", default=os.path.join(HERE, "curves.html"))
    args = ap.parse_args()

    csv_path = os.path.join(HERE, "curves.csv")
    subprocess.run([sys.executable, os.path.join(HERE, "parse_logs.py"),
                    "--logs", args.logs, "--out", csv_path], check=True)

    data = build_data(csv_path, args.logs, args.runs)
    json_path = os.path.join(HERE, "data.json")
    json.dump(data, open(json_path, "w"), separators=(",", ":"))
    print(f"wrote {json_path}")

    tpl_path = os.path.join(HERE, "template.html")
    tpl = open(tpl_path, encoding="utf-8").read()
    payload = json.dumps(data, separators=(",", ":"))
    if "</script" in payload:
        sys.exit("data contains '</script' and would break the page")
    page = tpl.replace("/*__DATA__*/", payload)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                 '<meta name="viewport" content="width=device-width,initial-scale=1">'
                 "</head><body>" + page + "</body></html>")
    print(f"wrote {args.out}")
    for r in data["runs"]:
        print(f"  {r['key']:<10} best {r['best']['mpjpe']:6.2f} @ep{r['best']['epoch']:<3} "
              f"spike {r['spike']:>2}/{len(r['epochs'])}  {r['duration']}")


if __name__ == "__main__":
    main()
