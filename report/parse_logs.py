#!/usr/bin/env python3
"""Parse sealpose training logs into a tidy CSV.

Reads logs/*_out.log and writes report/curves.csv with one row per epoch.
Standard library only, so it runs anywhere without extra installs.

Usage:
    python report/parse_logs.py                     # repo root assumed
    python report/parse_logs.py --logs DIR --out FILE
"""

import argparse
import csv
import os
import re
import sys

# "Epoch [33/50], Loss: 0.004497, Time taken: 33.58s, Early stopping: 0"
RE_EPOCH = re.compile(
    r"^Epoch \[(?P<epoch>\d+)/(?P<total>-?\d+)\], Loss: (?P<loss>[\d.eE+-]+), "
    r"Time taken: (?P<time_s>[\d.]+)s, Early stopping: (?P<early_stop>\d+)"
)
# "  Energy loss: -4.124E-05, E-diff: 25.467897, E-diff Ratio: 0.408"
RE_ENERGY = re.compile(
    r"^\s*Energy loss: (?P<energy>[\d.eE+-]+), E-diff: (?P<e_diff>[\d.eE+-]+), "
    r"E-diff Ratio: (?P<e_diff_ratio>[\d.eE+-]+)"
)
# "3dhp: Protocol #1   (MPJPE) overall average: 70.98 (mm)"
RE_P1 = re.compile(r"Protocol #1\s+\(MPJPE\) overall average: (?P<v>[\d.]+)")
RE_P2 = re.compile(r"Protocol #2\s+\(P-MPJPE\) overall average: (?P<v>[\d.]+)")
# "pck: 89.39, auc: 58.46"
RE_PCKAUC = re.compile(r"^pck: (?P<pck>[\d.]+), auc: (?P<auc>[\d.]+)")
# "Model saved at checkpoints/gcn-seal/, epoch 31"
RE_SAVED = re.compile(r"^Model saved at .*, epoch (?P<epoch>\d+)")
# "args     : --dataset 3dhp --batch_size 1024 ..."
RE_ARGS = re.compile(r"^args\s*:\s*(?P<args>.+)$")
RE_PARAMS = re.compile(r"^==> Number of parameters: (?P<v>[\d,]+)")
RE_PARAMS_LOSS = re.compile(r"^==> Number of parameters \(loss-net\): (?P<v>[\d,]+)")
RE_FNAME = re.compile(r"^(?P<run>.+)_(?P<jobid>\d+)_out\.log$")

FIELDS = [
    "run", "job_id", "task_net", "type", "lr", "energy_weight", "lr_loss",
    "epoch", "loss", "time_s", "early_stop",
    "mpjpe", "p_mpjpe", "pck", "auc",
    "energy", "e_diff", "e_diff_ratio", "saved",
]


def arg_value(args_str, flag, default=""):
    """Pull the value following `flag` out of a shell-style argument string."""
    toks = args_str.split()
    if flag in toks:
        i = toks.index(flag)
        if i + 1 < len(toks):
            return toks[i + 1]
    return default


def parse_log(path):
    """Return (meta, rows) for one *_out.log file."""
    fname = os.path.basename(path)
    m = RE_FNAME.match(fname)
    if not m:
        return None, []
    meta = {
        "run": m.group("run"),
        "job_id": m.group("jobid"),
        "args": "",
        "params": "",
        "params_loss": "",
    }

    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    saved_epochs = set()
    for line in lines:
        if not meta["args"]:
            a = RE_ARGS.match(line)
            if a:
                meta["args"] = a.group("args")
        p = RE_PARAMS.match(line)
        if p:
            meta["params"] = p.group("v")
        pl = RE_PARAMS_LOSS.match(line)
        if pl:
            meta["params_loss"] = pl.group("v")
        s = RE_SAVED.match(line)
        if s:
            saved_epochs.add(int(s.group("epoch")))

    args_str = meta["args"]
    common = {
        "run": meta["run"],
        "job_id": meta["job_id"],
        "task_net": arg_value(args_str, "--task_net", "linear"),
        "type": arg_value(args_str, "--type", "baseline"),
        "lr": arg_value(args_str, "--lr"),
        "energy_weight": arg_value(args_str, "--energy_weight"),
        "lr_loss": arg_value(args_str, "--lr_loss"),
    }

    rows, cur = [], None
    for line in lines:
        e = RE_EPOCH.match(line)
        if e:
            # A new Epoch header closes the previous block.
            if cur:
                rows.append(cur)
            cur = dict(common)
            cur.update(
                epoch=int(e.group("epoch")),
                loss=float(e.group("loss")),
                time_s=float(e.group("time_s")),
                early_stop=int(e.group("early_stop")),
                mpjpe="", p_mpjpe="", pck="", auc="",
                energy="", e_diff="", e_diff_ratio="",
            )
            cur["saved"] = int(cur["epoch"] in saved_epochs)
            continue
        if cur is None:
            continue
        en = RE_ENERGY.match(line)
        if en:
            cur["energy"] = float(en.group("energy"))
            cur["e_diff"] = float(en.group("e_diff"))
            cur["e_diff_ratio"] = float(en.group("e_diff_ratio"))
            continue
        p1 = RE_P1.search(line)
        if p1:
            cur["mpjpe"] = float(p1.group("v"))
            continue
        p2 = RE_P2.search(line)
        if p2:
            cur["p_mpjpe"] = float(p2.group("v"))
            continue
        pa = RE_PCKAUC.match(line)
        if pa:
            cur["pck"] = float(pa.group("pck"))
            cur["auc"] = float(pa.group("auc"))
    if cur:
        rows.append(cur)
    return meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs", help="directory holding *_out.log")
    ap.add_argument("--out", default="report/curves.csv", help="output CSV path")
    ap.add_argument("--exclude", default="smoke", help="skip runs whose name contains this")
    args = ap.parse_args()

    if not os.path.isdir(args.logs):
        sys.exit(f"log directory not found: {args.logs}")

    paths = sorted(
        os.path.join(args.logs, f)
        for f in os.listdir(args.logs)
        if f.endswith("_out.log")
    )
    all_rows = []
    for path in paths:
        meta, rows = parse_log(path)
        if not rows:
            continue
        if args.exclude and args.exclude in meta["run"]:
            print(f"skip   {meta['run']:<14} (excluded)")
            continue
        all_rows.extend(rows)
        best = min((r["mpjpe"] for r in rows if r["mpjpe"] != ""), default=None)
        print(
            f"parsed {meta['run']:<14} job {meta['job_id']}  "
            f"{len(rows):>3} epochs  best MPJPE {best}"
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {args.out}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
