#!/usr/bin/env python3
"""Figures of the scaling study (Milestone 6b) from docs/benchmark_results.json.

Writes docs/figures/scaling_*.png (150 dpi). Markers show medians, error bars min/max.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
STACK = {"spmv_local": ("local SpMV", "#2a78d6"), "halo_wait": ("halo wait", "#eb6834"),
         "spmv_ghost": ("A_ghost SpMV", "#1baf7a"), "allreduce": ("Allreduce", "#8b5cf6"),
         "other": ("vector ops + preconditioner + rest", "#a8a6a0")}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", FIG / name)


def err(med, lo, hi):
    med, lo, hi = map(np.asarray, (med, lo, hi))
    return np.vstack([np.maximum(med - lo, 0), np.maximum(hi - med, 0)])


def series(group, f):
    """sorted [(np, f(entry))]"""
    return sorted(((d["np"], f(d)) for d in group.values()), key=lambda t: t[0])


def main():
    R = json.loads((ROOT / "docs" / "benchmark_results.json").read_text())
    G = R["groups"]
    strong = [(f"strong_L{l}", f"L{l}") for l in (4, 5, 6) if f"strong_L{l}" in G]
    FIG.mkdir(parents=True, exist_ok=True)
    nps = [1, 2, 3, 4, 6, 8, 10]

    # (a) speedup, (b) efficiency
    for kind in ("speedup", "efficiency"):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        if kind == "speedup":
            ax.plot([1, 10], [1, 10], color=MUTED, linestyle="--", linewidth=1.2, label="ideal")
        else:
            ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2, label="ideal")
        for i, (g, label) in enumerate(strong):
            pts = series(G[g], lambda d: (d["derived"]["solve"]["speedup"],
                                          d["derived"]["solve"]["speedup_lo"],
                                          d["derived"]["solve"]["speedup_hi"]))
            p = np.array([t[0] for t in pts])
            m, lo, hi = (np.array([t[1][k] for t in pts]) for k in range(3))
            if kind == "efficiency":
                m, lo, hi = m / p, lo / p, hi / p
            ax.errorbar(p, m, yerr=err(m, lo, hi), color=COLORS[i], marker=MARKERS[i],
                        markersize=7, linewidth=2, capsize=3, label=label)
        style(ax, "MPI ranks p", "speedup $T_1/T_p$" if kind == "speedup" else "efficiency $S/p$",
              f"Strong scaling: {kind} (median solve time)")
        ax.set_xticks(nps)
        if kind == "efficiency":
            ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK)
        save(fig, f"scaling_strong_{kind}.png")

    # (c) stacked breakdown
    fig, axes = plt.subplots(1, len(strong), figsize=(4.6 * len(strong), 4.6), sharey=False)
    for ax, (g, label) in zip(np.atleast_1d(axes), strong):
        ents = sorted(G[g].values(), key=lambda d: d["np"])
        x = np.arange(len(ents))
        bottom = np.zeros(len(ents))
        parts = {k: np.array([e["summary"]["breakdown_mean"][k]["median"] for e in ents])
                 for k in ("spmv_local", "halo_wait", "spmv_ghost", "allreduce")}
        total = np.array([e["summary"]["breakdown_mean"]["total"]["median"] for e in ents])
        parts["other"] = total - sum(parts.values())
        for k, (name, c) in STACK.items():
            ax.bar(x, parts[k], bottom=bottom, color=c, label=name, width=0.7)
            bottom += parts[k]
        ax.set_xticks(x, [str(e["np"]) for e in ents])
        style(ax, "MPI ranks p", "time in solve [s] (mean over ranks)", f"{label}: solve breakdown")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper right")
    save(fig, "scaling_breakdown.png")

    # (d) setup phases for L6
    g6 = "strong_L6" if "strong_L6" in G else strong[-1][0]
    ents = sorted(G[g6].values(), key=lambda d: d["np"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    phases = [("read", "read (rank 0, serial)", "#2a78d6", "//"),
              ("partition", "partition (rank 0, serial)", "#eb6834", "//"),
              ("distribute", "distribute", "#1baf7a", ""),
              ("assembly", "assembly", "#8b5cf6", ""), ("bcs", "boundary conditions", "#d4a017", "")]
    x = np.arange(len(ents))
    bottom = np.zeros(len(ents))
    for k, name, c, h in phases:
        v = np.array([e["summary"]["phases"][k]["median"] for e in ents])
        ax.bar(x, v, bottom=bottom, color=c, hatch=h, edgecolor=SURFACE, label=name, width=0.7)
        bottom += v
    ax.set_xticks(x, [str(e["np"]) for e in ents])
    style(ax, "MPI ranks p", "time [s] (max over ranks)", f"{g6[-2:]}: setup phases")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK, loc="upper center", ncol=2)
    save(fig, "scaling_setup_phases.png")

    # (e) weak scaling
    weak = [(g, g[5:]) for g in ("weak_50k", "weak_200k") if g in G]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for i, (g, label) in enumerate(weak):
        ents = sorted(G[g].values(), key=lambda d: d["np"])
        p = [e["np"] for e in ents]
        for ax, key, scale in ((axes[0], "time_per_iteration", 1e3), (axes[1], "solve", 1),
                               (axes[2], "iterations", 1)):
            s = [e["summary"][key] for e in ents]
            m = np.array([a["median"] for a in s]) * scale
            lo = np.array([a["min"] for a in s]) * scale
            hi = np.array([a["max"] for a in s]) * scale
            ax.errorbar(p, m, yerr=err(m, lo, hi), color=COLORS[i], marker=MARKERS[i],
                        markersize=7, linewidth=2, capsize=3, label=f"{label} DOFs per rank")
    style(axes[0], "MPI ranks p", "time per iteration [ms]", "Weak scaling: time per iteration")
    style(axes[1], "MPI ranks p", "solve time [s]", "Total solve time")
    style(axes[2], "MPI ranks p", "CG iterations", "Iterations")
    for ax in axes:
        ax.set_xticks(nps)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK)
    save(fig, "scaling_weak.png")

    # (f) SpMV bandwidth vs triad
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    tri = R.get("triad", {})
    if tri:
        tp = sorted(int(k) for k in tri)
        ax.plot(tp, [tri[str(k)]["best_gbs"] for k in tp], color=MUTED, linestyle="--",
                linewidth=1.5, marker="x", markersize=7, label="STREAM-like triad (6a method)")
    for i, (g, label) in enumerate(strong):
        ents = sorted(G[g].values(), key=lambda d: d["np"])
        p = [e["np"] for e in ents]
        s = [e["summary"]["spmv_bandwidth_gbs"] for e in ents]
        m = np.array([a["median"] for a in s])
        ax.errorbar(p, m, yerr=err(m, [a["min"] for a in s], [a["max"] for a in s]),
                    color=COLORS[i], marker=MARKERS[i], markersize=7, linewidth=2, capsize=3,
                    label=f"SpMV, {label}")
    style(ax, "MPI ranks p", "effective bandwidth [GB/s]", "SpMV effective bandwidth vs triad")
    ax.set_xticks(nps)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    save(fig, "scaling_spmv_bandwidth.png")


if __name__ == "__main__":
    sys.exit(main())
