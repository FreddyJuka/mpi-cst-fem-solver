#!/usr/bin/env python3
"""Figures of the PETSc GAMG benchmark (Milestone 7b).

Reads docs/benchmark_gamg_results.json (GAMG, Jacobi anchor) and docs/benchmark_results.json
(Jacobi data of Milestone 6b). Writes docs/figures/gamg_*.png (150 dpi). Markers show medians,
error bars min/max.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_scaling as ps  # noqa: E402  (style helpers; sets the Agg backend)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = ps.ROOT
FIG = ps.FIG
COLORS, MARKERS = ps.COLORS, ps.MARKERS
JAC, GAM = "#8b5cf6", "#1baf7a"
NPS = [1, 2, 3, 4, 6, 8, 10]


def load():
    g = json.loads((ROOT / "docs" / "benchmark_gamg_results.json").read_text())["groups"]
    j = json.loads((ROOT / "docs" / "benchmark_results.json").read_text())["groups"]
    return g, j


def srt(group):
    return sorted(((d["np"], d) for k, d in group.items() if not k.startswith("anchor")),
                  key=lambda t: t[0])


def main():
    G, J = load()
    ps.FIG.mkdir(parents=True, exist_ok=True)
    levels = [4, 5, 6]

    # (a) time to solution vs DOFs, np = 1 and np = 8
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    for ax, p in zip(axes, (1, 8)):
        for name, src, field, color, mk, label in (
                ("Jacobi", J, "total", JAC, "o", "in-house CG + Jacobi (6b)"),
                ("GAMG", G, "total", GAM, "s", "PETSc CG + GAMG")):
            pre = "strong_L" if src is J else "gamg_strong_L"
            dofs, m, lo, hi = [], [], [], []
            for l in levels:
                d = src.get(f"{pre}{l}", {}).get(f"np{p}")
                if not d:
                    continue
                dofs.append(d["runs"][0]["dofs"])
                s = d["summary"][field]
                m.append(s["median"]); lo.append(s["min"]); hi.append(s["max"])
            ax.errorbar(dofs, m, yerr=ps.err(m, lo, hi), color=color, marker=mk, markersize=7,
                        linewidth=2, capsize=3, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
        ps.style(ax, "DOFs", "time to solution (total) [s]", f"p = {p}")
        ax.legend(fontsize=8, frameon=False, labelcolor=ps.INK)
    ps.save(fig, "gamg_time_to_solution.png")

    # (b) iterations vs DOFs with fitted slopes (np = 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for color, mk, src, pre, label in ((JAC, "o", J, "strong_L", "CG + Jacobi"),
                                       (GAM, "s", G, "gamg_strong_L", "CG + GAMG")):
        dofs = np.array([src[f"{pre}{l}"]["np1"]["runs"][0]["dofs"] for l in levels], float)
        its = np.array([src[f"{pre}{l}"]["np1"]["summary"]["iterations"]["median"] for l in levels])
        slope = np.polyfit(np.log(dofs), np.log(its), 1)
        ax.plot(dofs, its, color=color, marker=mk, markersize=7, linewidth=2,
                label=f"{label}: iterations ~ DOFs$^{{{slope[0]:.2f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ps.style(ax, "DOFs", "CG iterations (rtol 1e-10)", "Iterations vs problem size")
    ax.legend(fontsize=8, frameon=False, labelcolor=ps.INK)
    ps.save(fig, "gamg_iterations.png")

    # (c) GAMG strong speedup, total and solve-only
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, what, title in ((axes[0], "total", "total time"), (axes[1], "solve_only",
                                                               "solve-only (PC setup + KSP solve)")):
        ax.plot([1, 10], [1, 10], color=ps.MUTED, linestyle="--", linewidth=1.2, label="ideal")
        for i, l in enumerate(levels):
            pts = srt(G[f"gamg_strong_L{l}"])
            p = np.array([t[0] for t in pts])
            m, lo, hi = (np.array([t[1]["derived"][what][k] for t in pts])
                         for k in ("speedup", "speedup_lo", "speedup_hi"))
            ax.errorbar(p, m, yerr=ps.err(m, lo, hi), color=COLORS[i], marker=MARKERS[i],
                        markersize=7, linewidth=2, capsize=3, label=f"L{l}")
        ps.style(ax, "MPI ranks p", "speedup $T_1/T_p$", f"GAMG strong scaling: {title}")
        ax.set_xticks(NPS)
        ax.legend(fontsize=8, frameon=False, labelcolor=ps.INK)
    ps.save(fig, "gamg_strong_speedup.png")

    # (d) stacked phases for L6: Jacobi (left bar) vs GAMG (right bar) per p
    phases = [("read", "read (rank 0, serial)", "#2a78d6", "//"),
              ("partition", "partition (rank 0, serial)", "#eb6834", "//"),
              ("distribute", "distribute", "#1baf7a", ""),
              ("assembly", "assembly", "#8b5cf6", ""), ("bcs", "boundary conditions", "#d4a017", ""),
              ("petsc_matrix", "PETSc matrix conversion", "#7a4b2a", ""),
              ("pc_setup", "GAMG setup", "#c0392b", ""),
              ("ksp_solve", "CG iterations / Jacobi-CG solve", "#4b4a47", ""),
              ("rest", "postprocess + output", "#a8a6a0", "")]
    nps = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    w = 0.38

    def phase_values(src, lab, k):
        v = []
        for p in nps:
            s = src[f"np{p}"]["summary"]["phases"]
            if k == "rest":
                v.append(s["postprocess"]["median"] + s["output"]["median"])
            elif k == "ksp_solve" and lab == "J":
                v.append(s["solve"]["median"])
            elif k in ("petsc_matrix", "pc_setup") and lab == "J":
                v.append(0.0)
            else:
                v.append(s[k]["median"])
        return np.array(v)

    # left: Jacobi (J) and GAMG (G) side by side; right: GAMG alone on its own scale
    for ax, bars in ((axes[0], ((-w / 2 - 0.02, J["strong_L6"], "J"), (w / 2 + 0.02, G["gamg_strong_L6"], "G"))),
                     (axes[1], ((0.0, G["gamg_strong_L6"], "G"),))):
        for off, src, lab in bars:
            bottom = np.zeros(len(nps))
            for k, label, c, h in phases:
                v = phase_values(src, lab, k)
                ax.bar(np.arange(len(nps)) + off, v, w if len(bars) > 1 else 0.6, bottom=bottom,
                       color=c, hatch=h, edgecolor=ps.SURFACE,
                       label=label if (lab == "G" and ax is axes[1]) else None)
                bottom += v
            if len(bars) > 1:
                for xi, b in enumerate(bottom):
                    ax.text(xi + off, b, lab, ha="center", va="bottom", fontsize=8, color=ps.MUTED)
        ax.set_xticks(np.arange(len(nps)), [str(p) for p in nps])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ps.style(axes[0], "MPI ranks p   (J = Jacobi, G = GAMG)", "time [s] (median of per-phase max)",
             "L6: Jacobi vs GAMG, common scale")
    ps.style(axes[1], "MPI ranks p", "time [s] (median of per-phase max)", "L6: GAMG only, own scale")
    axes[1].legend(fontsize=7, frameon=False, labelcolor=ps.INK, ncol=2, loc="upper right")
    ps.save(fig, "gamg_phase_breakdown_L6.png")

    # (e) weak scaling with GAMG
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for i, tag in enumerate(("50k", "200k")):
        pts = srt(G[f"gamg_weak_{tag}"])
        p = [t[0] for t in pts]
        for ax, key, scale in ((axes[0], "iterations", 1), (axes[1], "solve_only", 1),
                               (axes[2], "total", 1)):
            s = [t[1]["summary"][key] for t in pts]
            m = np.array([a["median"] for a in s]); lo = np.array([a["min"] for a in s])
            hi = np.array([a["max"] for a in s])
            ax.errorbar(p, m, yerr=ps.err(m, lo, hi), color=COLORS[i], marker=MARKERS[i],
                        markersize=7, linewidth=2, capsize=3, label=f"{tag} DOFs per rank")
    ps.style(axes[0], "MPI ranks p", "CG iterations", "GAMG weak scaling: iterations")
    ps.style(axes[1], "MPI ranks p", "PC setup + KSP solve [s]", "Solve-only time")
    ps.style(axes[2], "MPI ranks p", "total time [s]", "Total time")
    for ax in axes:
        ax.set_xticks(NPS)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=ps.INK)
    ps.save(fig, "gamg_weak.png")


if __name__ == "__main__":
    sys.exit(main())
