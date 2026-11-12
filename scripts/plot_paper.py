#!/usr/bin/env python3
"""Figures of the paper, from the result files already committed (nothing is re-run).

  python scripts/plot_paper.py            -> results/figures/{fig1_quality_time,fig2_training,fig3_win_tie_loss}.{pdf,png}
                                              results/figures/win_tie_loss.csv

Fig. 1  Re-rostering on Burke & Curtois Instances 5, 8, 12 (big disruptions on 8 and 12): mean J of
        the 16 held-out contexts against the mean total response time T_infer, for every method at
        the same T_infer (Eqs. (28), (42)); the dashed line is the CP-SAT reference (60 s).
Fig. 2  Training curves: construction on 40x28x3, the constructive re-rostering policy on Instance 5
        (with and without the prior towards x°), the learned LNS on Instances 5, 8 and 12.
Fig. 3  Win / tie / loss of the learned methods against the repair of x° (no learning), per context
        (16 per row), at the same time budget.

Colours: the validated categorical palette of the dataviz skill (slots 1-5 in fixed order); every
method also has its own marker, so identity never rests on colour alone.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

R = Path("results")
OUT = R / "figures"

# one entry per method, fixed colour and marker (slots 1-5 of the palette)
STYLE = {
    "repair_x0":    ("Repair of x° (no learning)",          "#2a78d6", "o"),
    "cp_hint":      ("CP-SAT hinted with x°",               "#eb6834", "s"),
    "cp_scratch":   ("CP-SAT from scratch",                 "#1baf7a", "^"),
    "constructive": ("Constructive RL policy + repair",     "#eda100", "D"),
    "learned_lns":  ("Learned LNS (RL)",                    "#e87ba4", "v"),
}
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"
WIN, TIE, LOSS = "#2a78d6", "#c9c8c1", "#eb6834"


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


# ----------------------------------------------------------------------------- data
def repair_curves(name):
    """From results/repair/<name>/results.csv: (T, J) of the source and of both CP-SAT baselines,
    one point per tau_cp > 0, means over the contexts (J over the feasible ones)."""
    by = defaultdict(list)
    for r in rows(R / "repair" / name / "results.csv"):
        if num(r["tau_cp"]) > 0:
            by[num(r["tau_cp"])].append(r)
    out = {"source": [], "cp_scratch": [], "cp_hint": []}
    for tau in sorted(by):
        g = by[tau]
        T = np.mean([num(r["infer_seconds"]) for r in g])
        out["source"].append((T, np.nanmean([num(r["J"]) for r in g if r["feasible"] == "True"])))
        for key, col, ok in (("cp_scratch", "J_cp_scratch", "feasible_cp_scratch"),
                             ("cp_hint", "J_cp_hint", "feasible_cp_hint")):
            vals = [num(r[col]) for r in g if r[ok] == "True"]
            if vals:
                out[key].append((T, float(np.mean(vals))))
    return out


def lns_curves(name):
    """From results/lns/<name>/results.csv: (T, J) per method, one point per K."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows(R / "lns" / name / "results.csv"):
        by[r["method"]][int(r["K"])].append(r)
    out = {}
    for m, ks in by.items():
        out[m] = [(np.mean([num(r["budget"]) for r in g]),
                   float(np.mean([num(r["J"]) for r in g if r["feasible"] == "True"]))) for _, g in sorted(ks.items())]
    return out


def reference(name, kind):
    path = R / kind / name / "results.csv"
    col = "J_cp" if kind == "repair" else "J_cp_ref"
    seen = {}
    for r in rows(path):
        seen[r["instance" if kind == "repair" else "context"]] = num(r[col])
    return float(np.mean(list(seen.values())))


# ----------------------------------------------------------------------------- figure 1
def fig1():
    panels = [
        ("Instance 5 (16×28×2), small disruptions", {
            "repair_x0": repair_curves("bc5-rr-published-anytime")["source"],
            "cp_hint": repair_curves("bc5-rr-published-anytime")["cp_hint"],
            "cp_scratch": repair_curves("bc5-rr-published-anytime")["cp_scratch"],
            "constructive": repair_curves("bc5-rr-prior5-anytime")["source"],
            "learned_lns": lns_curves("bc5-lns")["learned_lns"],
        }, reference("bc5-rr-published-anytime", "repair")),
        ("Instance 8 (30×28×4), big disruptions", {
            "repair_x0": repair_curves("bc8-big-p5-published-anytime")["source"],
            "cp_hint": repair_curves("bc8-big-p5-published-anytime")["cp_hint"],
            "cp_scratch": repair_curves("bc8-big-p5-published-anytime")["cp_scratch"],
            "constructive": repair_curves("bc8-big-p5-bestckpt-anytime")["source"],
            "learned_lns": lns_curves("bc8-lns2")["learned_lns"],
        }, reference("bc8-big-p5-published-anytime", "repair")),
        ("Instance 12 (60×28×10), big disruptions", {
            m: lns_curves("bc12-lns")[m] for m in ("repair_x0", "cp_hint", "cp_scratch", "learned_lns")
        }, reference("bc12-lns", "lns")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (title, series, ref) in zip(axes, panels):
        ys = [J for pts in series.values() for _, J in pts]
        top = max(J for J in ys if J <= 1.12 * ref)          # far-off points are drawn on the top edge
        lo = min(ys + [ref])
        span = top - lo
        ax.set_ylim(lo - 0.04 * span, top + 0.06 * span)
        edge = top + 0.06 * span
        for key in STYLE:
            if key not in series or not series[key]:
                continue
            label, color, marker = STYLE[key]
            T, J = zip(*series[key])
            ax.plot(T, J, color=color, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
            for t, j in zip(T, J):
                if j > edge:
                    ax.plot([t], [edge], marker=marker, color=color, markersize=6, markeredgecolor="white",
                            markeredgewidth=1.2, clip_on=False, zorder=4)
                    ax.annotate(f"↑ {j:.0f}", (t, edge), xytext=(5, -2), textcoords="offset points",
                                ha="left", va="top", fontsize=7.5, color=MUTED)
        ax.axhline(ref, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
        ax.annotate("CP-SAT reference, 60 s", (0.02, ref), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="left", va="bottom", fontsize=8, color=MUTED)
        ax.set_xscale("log")
        T_all = [t for pts in series.values() for t, _ in pts]
        ax.set_xlim(min(T_all) / 1.25, max(T_all) * 1.25)
        narrow = max(T_all) / min(T_all) < 10
        ticks = [0.5, 0.7, 1, 1.4, 2, 2.8, 4] if narrow else [0.1, 0.2, 0.5, 1, 2, 5]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Total response time T_infer (s, log scale)", fontsize=9)
        style_axes(ax)
    axes[0].set_ylabel("Mean J over 16 contexts (lower is better)", fontsize=9)
    handles = [plt.Line2D([], [], color=c, marker=mk, linewidth=2, markersize=6, markeredgecolor="white",
                          label=lb) for lb, c, mk in STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig1_quality_time")


# ----------------------------------------------------------------------------- figure 2
def fig2():
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 3.7))
    OTHER = ("Other variant", MUTED, "s")

    def curve(ax, path, x, y, style, label=None, scale=1.0, dashed=False):
        rr = rows(path)
        _, color, marker = style
        ax.plot([num(r[x]) * scale for r in rr], [num(r[y]) for r in rr], color=color, linewidth=2,
                linestyle=(0, (5, 2)) if dashed else "-", marker=marker, markersize=4,
                markevery=max(1, len(rr) // 8), label=label)
        return rr

    def hline(ax, y, text, color=MUTED):
        ax.axhline(y, color=color, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.annotate(text, (0.98, y), xycoords=("axes fraction", "data"), xytext=(0, -3),
                    textcoords="offset points", ha="right", va="top", fontsize=8, color=MUTED)

    # (a) construction, 40x28
    ax = axes[0]
    rr = curve(ax, R / "train" / "mac-40x28-F-init" / "progress.csv", "steps", "J_det", STYLE["constructive"],
               scale=1e-6)
    hline(ax, num(rr[0]["J_cp"]), "CP-SAT")
    ax.set_yscale("log")
    ax.set_title("(a) Construction, 40×28×3\npolicy alone (no repair)", fontsize=10, loc="left")

    # (b) constructive re-rostering, Instance 5
    ax = axes[1]
    curve(ax, R / "train" / "bc5-rr" / "progress.csv", "steps", "J_det", OTHER, "No prior", 1e-6, dashed=True)
    curve(ax, R / "train" / "bc5-rr-prior5" / "progress.csv", "steps", "J_det", STYLE["constructive"],
          "Prior b0 = 5 (Fig. 1)", 1e-6)
    x0 = np.mean([num(r["J"]) for r in rows(R / "repair" / "bc5-rr-published-anytime" / "results.csv")
                  if num(r["tau_cp"]) == 0])                 # all contexts, as J_det averages all
    hline(ax, x0, "x° without the absent", STYLE["repair_x0"][1])
    ax.set_title("(b) Constructive RL, Instance 5\npolicy alone (no repair)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)

    # (c), (d), (e) learned LNS
    for ax, names, title in ((axes[2], (("bc5-lns", STYLE["learned_lns"], "Employees and days (Fig. 1)", False),),
                              "(c) Learned LNS (employees and days), Instance 5"),
                             (axes[3], (("bc8-lns1", OTHER, "Employees only", True),
                                        ("bc8-lns2", STYLE["learned_lns"], "Employees and days (Fig. 1)", False)),
                              "(d) Learned LNS, Instance 8"),
                             (axes[4], (("bc12-lns", STYLE["learned_lns"], "Employees and days (Fig. 1)", False),),
                              "(e) Learned LNS (employees and days), Instance 12")):
        start = None
        for name, style, label, dashed in names:
            rr = curve(ax, R / "train_lns" / name / "progress.csv", "steps", "J_det", style, label, 1e-3, dashed)
            start = num(rr[0]["J_start"])
        hline(ax, start, "start: x° without the absent", STYLE["repair_x0"][1])
        ax.set_title(title + "\nafter K = 8 neighbourhood steps", fontsize=10, loc="left")
        if len(names) > 1:                                   # one series: the title names it
            ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.9), labelcolor=INK)
    axes[0].set_ylabel("Mean J on held-out contexts", fontsize=9)
    for ax in axes[:2]:
        ax.set_xlabel("Training steps (millions)", fontsize=9)
    for ax in axes[2:]:
        ax.set_xlabel("Training steps (thousands)", fontsize=9)
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    save(fig, "fig2_training")


# ----------------------------------------------------------------------------- figure 3
def head_to_head(policy, baseline):
    """Win / tie / loss of ``policy`` against ``baseline`` (repair results), per tau_cp > 0."""
    a = {(r["instance"], r["tau_cp"]): r for r in rows(R / "repair" / policy / "results.csv")}
    b = {(r["instance"], r["tau_cp"]): r for r in rows(R / "repair" / baseline / "results.csv")}
    out = defaultdict(lambda: [0, 0, 0])
    for k, ra in a.items():
        if num(k[1]) <= 0 or k not in b:
            continue
        rb = b[k]
        fa, fb = ra["feasible"] == "True", rb["feasible"] == "True"
        if fa != fb:
            i = 0 if fa else 2
        elif not fa or abs(num(ra["J"]) - num(rb["J"])) <= 1e-9:
            i = 1
        else:
            i = 0 if num(ra["J"]) < num(rb["J"]) else 2
        out[num(k[1])][i] += 1
    return out


def fig3():
    table = []
    for inst, policy, baseline in (("Instance 5", "bc5-rr-prior5-anytime", "bc5-rr-published-anytime"),
                                   ("Instance 8", "bc8-big-p5-bestckpt-anytime", "bc8-big-p5-published-anytime")):
        for tau, wtl in sorted(head_to_head(policy, baseline).items()):
            table.append((inst, "Constructive RL", f"τ_cp = {tau:g} s", *wtl))
    for inst, name in (("Instance 5", "bc5-lns"), ("Instance 8", "bc8-lns2"), ("Instance 12", "bc12-lns")):
        summ = json.loads((R / "lns" / name / "run.json").read_text())["summary"]
        for K, s in sorted(summ.items(), key=lambda kv: int(kv[0])):
            table.append((inst, "Learned LNS", f"K = {K} ({s['T_infer']:.2f} s)", *s["vs_repair_x0"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "win_tie_loss.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "method", "budget", "wins", "ties", "losses"])
        w.writerows(table)

    groups = [(t[0], t[1]) for t in table]
    y, pos = [], 0.0
    for k, g in enumerate(groups):
        if k and g != groups[k - 1]:
            pos += 0.6                                       # gap between instance/method groups
        y.append(-pos)
        pos += 1
    fig, ax = plt.subplots(figsize=(8.5, 0.3 * pos + 1.1))
    for yi, (inst, method, budget, wi, ti, lo) in zip(y, table):
        left = 0
        for val, color in ((wi, WIN), (ti, TIE), (lo, LOSS)):
            if val:
                ax.barh(yi, val, left=left, color=color, height=0.74, edgecolor="white", linewidth=2)
                cx = left + val / 2
                if abs(cx - 0.6 * 16) < 0.45:                # keep the count off the criterion line
                    cx += 0.6
                ax.text(cx, yi, str(val), ha="center", va="center", fontsize=7.5,
                        color="white" if color != TIE else INK)
            left += val
    ax.set_yticks(y)
    ax.set_yticklabels([f"{inst} · {method} · {budget}" for inst, method, budget, *_ in table], fontsize=8)
    ax.set_xlim(0, 16)
    ax.set_ylim(y[-1] - 0.7, 0.7)
    ax.set_xlabel("Contexts (of 16): win / tie / loss against the repair of x° at the same time", fontsize=9)
    ax.axvline(0.6 * 16, color=INK, linestyle=(0, (4, 3)), linewidth=1, zorder=3)
    ax.annotate("success criterion:\n60% wins", (0.6 * 16, 0.7), xytext=(4, 2), textcoords="offset points",
                ha="left", va="bottom", fontsize=7.5, color=MUTED, annotation_clip=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (WIN, TIE, LOSS)]
    ax.legend(handles, ["Win", "Tie", "Loss"], ncol=3, frameon=False, fontsize=8, loc="lower left",
              bbox_to_anchor=(0, 1.0), labelcolor=INK, handlelength=1.2)
    style_axes(ax)
    ax.grid(False)
    fig.tight_layout()
    save(fig, "fig3_win_tie_loss")
    return table


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"written {OUT / name}.pdf / .png")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Lato", "Helvetica", "Arial",
                                                                          "DejaVu Sans"]})
    fig1()
    fig2()
    for row in fig3():
        print("  ", *row)
