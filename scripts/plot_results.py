#!/usr/bin/env python3
"""Figures for the paper, as vector PDF (plus a PNG preview), from recorded results.

  python scripts/plot_results.py cp results/cp_baseline/<run>

cp: from <run>/results.csv (scripts/solve_cp.py), two bar charts in <run>/figures/:
  gap_to_best_known.pdf   (J_CP - best known) / best known per instance; "opt" marks
                          instances the solver proved optimal
  solve_time.pdf          CP-SAT wall-clock time per instance, log scale, with the
                          time limit as a dashed line

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    cp = sub.add_parser("cp", help="CP-SAT baseline: gap to best known and solve time")
    cp.add_argument("run", type=Path, help="results/cp_baseline/<run>")
    args = ap.parse_args()
    if args.what == "cp":
        for p in plot_cp(args.run):
            print(p)


if __name__ == "__main__":
    main()
