#!/usr/bin/env python3
"""Markdown tables of the PETSc GAMG benchmark (Milestone 7b) from the two results JSON files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G = json.loads((ROOT / "docs" / "benchmark_gamg_results.json").read_text())
J = json.loads((ROOT / "docs" / "benchmark_results.json").read_text())["groups"]
GG = G["groups"]
NPS = [1, 2, 3, 4, 6, 8, 10]
PH = ("read", "partition", "distribute", "assembly", "bcs", "petsc_matrix", "pc_setup", "ksp_solve",
      "postprocess", "output", "total")


def f(x, n=3):
    return f"{x:.{n}g}"


def pr(title, head, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")


def flag(d):
    return " ⚠" if d["summary"]["noisy"] else ""


for l in (4, 5, 6):
    rows = []
    for p in NPS:
        d = GG[f"gamg_strong_L{l}"][f"np{p}"]
        s, dv = d["summary"], d["derived"]
        rows.append([p, int(s["iterations"]["median"]), f(s["total"]["median"]),
                     f(s["total"]["cv"] * 100, 2) + flag(d), f(s["solve_only"]["median"]),
                     f(s["solve_only"]["cv"] * 100, 2), f(dv["total"]["speedup"]),
                     f(dv["total"]["efficiency"]),
                     f(dv["total"]["karp_flatt"], 2) if p > 1 else "–",
                     f(dv["solve_only"]["speedup"]), f(dv["solve_only"]["efficiency"])])
    pr(f"GAMG strong scaling, L{l} ({GG[f'gamg_strong_L{l}']['np1']['runs'][0]['dofs']} DOFs)",
       ["p", "its", "total [s]", "CV total %", "solve-only [s]", "CV solve %", "S total", "E total",
        "Karp–Flatt (total)", "S solve", "E solve"], rows)

for tag in ("50k", "200k"):
    rows = []
    for p in NPS:
        d = GG[f"gamg_weak_{tag}"][f"np{p}"]
        s, dv = d["summary"], d["derived"]
        rows.append([p, int(round(dv["dofs_per_rank"] * p)), int(s["iterations"]["median"]),
                     f(s["solve_only"]["median"]), f(s["solve_only"]["cv"] * 100, 2) + flag(d),
                     f(s["total"]["median"]), f(s["total"]["cv"] * 100, 2),
                     f(dv["weak_efficiency_solve"]), f(dv["weak_efficiency_total"])])
    pr(f"GAMG weak scaling, {tag} DOFs per rank",
       ["p", "DOFs", "its", "solve-only [s]", "CV solve %", "total [s]", "CV total %",
        "weak eff. (solve-only)", "weak eff. (total)"], rows)

rows = []
for l in (4, 5, 6):
    for p in NPS:
        d = GG[f"gamg_strong_L{l}"][f"np{p}"]
        v = d["derived"].get("vs_jacobi")
        s = d["summary"]
        if v:
            rows.append([f"L{l}", p, int(v["jacobi_iterations"]), int(s["iterations"]["median"]),
                         f(v["jacobi_total"]), f(s["total"]["median"]), f(v["speedup_total"]),
                         f(v["jacobi_solve"]), f(s["solve_only"]["median"]),
                         f(v["speedup_solve"])])
pr("Time to solution, GAMG vs Jacobi (Jacobi: 6b medians)",
   ["mesh", "p", "its Jacobi", "its GAMG", "total Jacobi [s]", "total GAMG [s]", "speedup total",
    "solve Jacobi [s]", "solve-only GAMG [s]", "speedup solve"], rows)

rows = []
for name, label in (("gamg_strong_L4", "L4"), ("gamg_strong_L5", "L5"), ("gamg_strong_L6", "L6"),
                    ("gamg_weak_50k", "weak 50k"), ("gamg_weak_200k", "weak 200k")):
    for p in NPS:
        d = GG[name][f"np{p}"]
        gf = d["summary"]["serial_fraction"]["median"]
        dv = d.get("derived", {})
        jf = dv.get("jacobi_serial_fraction") or dv.get("vs_jacobi", {}).get("jacobi_serial_fraction")
        rows.append([label, p, f(gf * 100, 3), f(jf * 100, 3) if jf is not None else "–"])
pr("Fraction of total time in rank-0 serial phases (read + partition), %",
   ["case", "p", "GAMG", "Jacobi (6b)"], rows)

rows = [[k, f(v["anchor"]), f(v["ref_6b"]), f(v["ratio"]), "FLAG" if v["flagged"] else "ok"]
        for k, v in G["anchor_check"].items()]
pr("Anchor: in-house CG + Jacobi on L5 (3 repetitions) vs 6b medians",
   ["config", "anchor median [s]", "6b median [s]", "ratio", ""], rows)

rows = [[name, k, f(d["summary"]["total"]["cv"] * 100, 2), f(d["summary"]["solve_only"]["cv"] * 100, 2)]
        for name, dd in GG.items() for k, d in dd.items() if d["summary"]["noisy"]]
pr("Configurations with CV > 5 % (total or solve-only)",
   ["group", "key", "CV total %", "CV solve-only %"], rows)

for name in ("gamg_strong_L4", "gamg_strong_L5", "gamg_strong_L6", "gamg_weak_50k", "gamg_weak_200k"):
    rows = []
    for p in NPS:
        ph = GG[name][f"np{p}"]["summary"]["phases"]
        rows.append([p] + [f"{f(ph[k]['median'])} ({f(ph[k]['min'])}–{f(ph[k]['max'])})" for k in PH])
    pr(f"Phases {name[5:]}: median (min–max) [s]", ["p", *PH], rows)
