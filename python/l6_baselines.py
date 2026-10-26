#!/usr/bin/env python3
"""L6 strong-scaling speedups against two p = 1 baselines (Milestone 7c).

The L6 GAMG p = 1 times are bimodal (see docs/BENCHMARKS.md). "median" is the median of all five
repetitions (the baseline of the main tables); "fast" is the median of the repetitions with
total < FAST_LIMIT s (the fast mode, two of five runs). Prints the markdown tables used in the docs.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAST_LIMIT = 24.0  # s; the 5 runs are 19.7, 20.2 | 28.1, 28.3, 28.6 (gap 20.2-28.1)


def load():
    return json.loads((ROOT / "docs" / "benchmark_gamg_results.json").read_text())["groups"]


def baselines(G):
    runs = G["gamg_strong_L6"]["np1"]["runs"]
    fast = [r for r in runs if r["total"] < FAST_LIMIT]
    return dict(
        median=dict(total=statistics.median(r["total"] for r in runs),
                    solve=statistics.median(r["solve_only"] for r in runs)),
        fast=dict(total=statistics.median(r["total"] for r in fast),
                  solve=statistics.median(r["solve_only"] for r in fast), n=len(fast)))


def rows(G):
    b = baselines(G)
    J = json.loads((ROOT / "docs" / "benchmark_results.json").read_text())["groups"]
    g6 = G["gamg_strong_L6"]
    out = []
    for key in sorted(g6, key=lambda k: g6[k]["np"]):
        d = g6[key]
        tm, sm = d["summary"]["total"]["median"], d["summary"]["solve_only"]["median"]
        out.append(dict(np=d["np"], total=tm, solve=sm,
                        s_tot_med=b["median"]["total"] / tm, s_tot_fast=b["fast"]["total"] / tm,
                        s_sol_med=b["median"]["solve"] / sm, s_sol_fast=b["fast"]["solve"] / sm))
    return b, out, J["strong_L6"]["np1"]["summary"]


if __name__ == "__main__":
    G = load()
    b, out, jac = rows(G)
    print(json.dumps(b, indent=1))
    print("| p | total [s] | S total (median baseline) | S total (fast baseline) | solve-only [s] "
          "| S solve (median baseline) | S solve (fast baseline) |\n|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['np']} | {r['total']:.1f} | {r['s_tot_med']:.2f} | {r['s_tot_fast']:.2f} | "
              f"{r['solve']:.2f} | {r['s_sol_med']:.2f} | {r['s_sol_fast']:.2f} |")
    print(list(jac))
    jt, js = jac["total"]["median"], jac["solve"]["median"]
    for k in ("median", "fast"):
        print(k, "Jacobi/GAMG p=1: total", jt / b[k]["total"], "solve", js / b[k]["solve"])
