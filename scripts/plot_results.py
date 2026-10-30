#!/usr/bin/env python3
"""Figures for the paper, as vector PDF (plus a PNG preview), from recorded results.

  python scripts/plot_results.py cp results/cp_baseline/<run>
  python scripts/plot_results.py train results/train/<run>

cp: from <run>/results.csv (scripts/solve_cp.py), two bar charts in <run>/figures/:
  gap_to_best_known.pdf   (J_CP - best known) / best known per instance; "opt" marks
                          instances the solver proved optimal
  solve_time.pdf          CP-SAT wall-clock time per instance, log scale, with the
                          time limit as a dashed line

train: from <run>/progress.csv and <run>/results.csv (scripts/train.py), in <run>/figures/:
  learning_curve.pdf      mean held-out J against training steps (deterministic policy and
                          best of k samples), CP-SAT as a dashed reference, log scale
  feasibility_curve.pdf   share of held-out rosters feasible under the published profile
  final_gap.pdf           J_policy - J_CP per held-out instance at the end (absolute: the
                          optimum can be near 0, where a relative gap means nothing)

If an instance appears more than once in results.csv, its last row is used.
Single-column width (3.4 in), 8 pt sans text, one colour per series; titles
belong in the LaTeX caption, not in the figure.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

# palette (dataviz reference instance): series slot 1, ink, chrome
SERIES = "#2a78d6"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
WIDTH, HEIGHT = 3.4, 2.1


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    last = {}
    for r in rows:
        last[r["instance"]] = r
    key = lambda r: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", r["instance"])]  # noqa: E731
    return sorted(last.values(), key=key)


def num(v):
    return None if v in ("", "None", None) else float(v)


def short(name: str) -> str:
    m = re.fullmatch(r"Instance(\d+)", name)
    return m.group(1) if m else name


def save(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.pdf", out / f"{stem}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def plot_cp(run: Path) -> list[Path]:
    rows = read_rows(run / "results.csv")
    if not rows:
        raise SystemExit(f"{run}/results.csv has no rows")
    style()
    labels = [short(r["instance"]) for r in rows]
    xs = range(len(rows))
    out = run / "figures"
    written = []

    # 1. gap to the best known value
    gaps = [num(r["gap_to_best_known"]) for r in rows]
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    vals = [100 * g if g is not None else 0.0 for g in gaps]
    ax.bar(xs, vals, width=0.6, color=SERIES, linewidth=0)
    top = max(max(vals), 0.25)          # room for the labels even when every gap is 0
    for x, r, g, v in zip(xs, rows, gaps, vals):
        if r["status"] == "OPTIMAL":
            text = "opt"
        elif g is None:
            text = "none"
        else:
            text = f"{v:.1f}%"
        ax.text(x, max(v, 0) + 0.02 * top, text, ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xticks(list(xs), labels)
    ax.set_xlabel("Instance")
    ax.set_ylabel("Gap to best known (%)")
    ax.set_ylim(min(0, min(vals) * 1.15), top * 1.3)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.text(1.0, 1.02, "opt = proven optimal", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color=MUTED)
    written += save(fig, out, "gap_to_best_known")

    # 2. solve time, log scale, with the time limit
    times = [num(r["solve_seconds"]) or 0.0 for r in rows]
    limit = num(rows[0]["time_limit"])
    floor = 0.01
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    ax.bar(xs, [max(t, floor) for t in times], width=0.6, color=SERIES, linewidth=0, bottom=0)
    ax.set_yscale("log")
    ax.set_ylim(floor, (limit or max(times)) * 4)
    if limit:
        ax.axhline(limit, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        # left end, above the line: the short (solved) instances come first, so it is clear there
        ax.text(-0.4, limit * 1.15, f"time limit ({limit:g} s)", ha="left", va="bottom",
                fontsize=6.5, color=MUTED)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for x, r, t in zip(xs, rows, times):
        if r["status"] == "OPTIMAL":
            ax.text(x, max(t, floor) * 1.25, "opt", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xticks(list(xs), labels)
    ax.set_xlabel("Instance")
    ax.set_ylabel("CP-SAT time (s)")
    written += save(fig, out, "solve_time")
    return written


SERIES_2 = "#eb6834"      # categorical slot 2


def plot_train(run: Path) -> list[Path]:
    """Learning curve, feasibility curve and final gaps of a training run (scripts/train.py)."""
    style()
    out = run / "figures"
    written = []
    prog_path = run / "progress.csv"
    if prog_path.exists():
        with prog_path.open(newline="") as fh:
            prog = list(csv.DictReader(fh))
        steps = [float(r["steps"]) for r in prog]
        det = [float(r["J_det"]) for r in prog]
        sto = [float(r["J_stoch"]) for r in prog]
        cp = float(prog[-1]["J_cp"])
        k = "k"
        res_path = run / "results.csv"
        if res_path.exists():
            with res_path.open(newline="") as fh:
                first = next(csv.DictReader(fh), None)
            k = first["k"] if first else k

        # 1. mean held-out J against training steps, CP-SAT as the reference line
        fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
        ax.plot(steps, det, color=SERIES, linewidth=1.5, label="policy (deterministic)")
        ax.plot(steps, sto, color=SERIES_2, linewidth=1.5, label=f"policy (best of {k})")
        ax.axhline(cp, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), label="CP-SAT")
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Mean J, held-out instances")
        ax.set_yscale("log")
        lo, hi = min(det + sto + [cp]), max(det + sto + [cp])
        ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
                 if lo / 1.5 <= t <= hi * 1.5]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(lo / 1.6, hi * 1.3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
        # above the axes, in one row: the curves can take any shape, so inside they would collide
        ax.legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                  handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
        written += save(fig, out, "learning_curve")

        # 2. share of feasible rosters against training steps
        fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
        ax.plot(steps, [100 * float(r["feasible_det"]) for r in prog], color=SERIES, linewidth=1.5,
                label="policy (deterministic)")
        ax.plot(steps, [100 * float(r["feasible_stoch"]) for r in prog], color=SERIES_2, linewidth=1.5,
                label=f"policy (best of {k})")
        ax.set_ylim(-3, 103)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Feasible rosters (%)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
        ax.legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                  handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
        written += save(fig, out, "feasibility_curve")

    res_path = run / "results.csv"
    if res_path.exists():
        with res_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        # 3. final J - J_CP per held-out instance, two series side by side. Absolute, not relative:
        # the optimum can be close to 0, where a relative gap is meaningless
        fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
        xs = np.arange(len(rows))
        w = 0.38
        g_det = [float(r["J_det"]) - float(r["J_cp"]) for r in rows]
        g_best = [float(r["J_best_of_k"]) - float(r["J_cp"]) for r in rows]
        ax.bar(xs - w / 2 - 0.01, g_det, width=w, color=SERIES, linewidth=0, label="deterministic")
        ax.bar(xs + w / 2 + 0.01, g_best, width=w, color=SERIES_2, linewidth=0,
               label=f"best of {rows[0]['k']}")
        top = max(g_det + g_best + [1.0])
        for x, r, gd, gb in zip(xs, rows, g_det, g_best):
            if r["feasible_det"] != "True":
                ax.text(x - w / 2, max(gd, 0) + 0.02 * top, "infeas.", ha="center", va="bottom",
                        fontsize=5, color=INK_2, rotation=90)
            if r["feasible_best_of_k"] != "True":
                ax.text(x + w / 2, max(gb, 0) + 0.02 * top, "infeas.", ha="center", va="bottom",
                        fontsize=5, color=INK_2, rotation=90)
        ax.set_ylim(min(0, min(g_det + g_best) * 1.1), top * 1.25)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xticks(xs, [str(i + 1) for i in xs])
        ax.set_xlabel("Held-out instance")
        ax.set_ylabel("J − J of CP-SAT")
        ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
        written += save(fig, out, "final_gap")
    if not written:
        raise SystemExit(f"{run} has neither progress.csv nor results.csv")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    cp = sub.add_parser("cp", help="CP-SAT baseline: gap to best known and solve time")
    cp.add_argument("run", type=Path, help="results/cp_baseline/<run>")
    tr = sub.add_parser("train", help="training run: learning curve, feasibility, final gaps")
    tr.add_argument("run", type=Path, help="results/train/<run>")
    args = ap.parse_args()
    if args.what == "cp":
        for p in plot_cp(args.run):
            print(p)
    elif args.what == "train":
        for p in plot_train(args.run):
            print(p)


if __name__ == "__main__":
    main()
