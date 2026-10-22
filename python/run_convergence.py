#!/usr/bin/env python3
"""Kirsch mesh-convergence study (SPEC section 8).

Runs cstfem_solve on the Kirsch meshes (levels 0-5 by default) on the distributed path,
plane stress (figures + table) and plane strain (table), reads timings.json and the CSV
files, computes the mesh-size measures, fits convergence rates, writes
docs/verification_results.json and the figures in docs/figures/.

Mesh size per level: h_max = longest triangle edge, h_eff = sqrt(total area / #triangles).
Rates: least-squares slope of log(error) vs log(h_eff) over all levels and over levels
2-5, plus consecutive-level rates log(e_l / e_l+1) / log(h_l / h_l+1).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mesh import parse_msh  # noqa: E402  (independent .msh parser)

ROOT = Path(__file__).resolve().parent.parent
SIGMA_INF, A = 100e6, 0.1
E, NU, T = 210e9, 0.3, 0.01
METRICS = [  # key in timings.json, label, expected rate
    ("l2_rel_displacement_error", "displacement, L2", 2.0),
    ("energy_rel_error", "energy norm", 1.0),
    ("l2_rel_stress_error", "stress, L2 (centroid)", 1.0),
]
# Reference categorical palette, fixed order (slots 1-3), plus text/grid tokens.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def mesh_measures(path):
    _, coords, tris, _ = parse_msh(path)
    tags = {t: i for i, t in enumerate(coords)}
    xy = np.array([coords[t][:2] for t in coords])
    conn = np.array([[tags[n] for n in c] for c, _ in tris])
    p = xy[conn]  # (ne, 3, 2)
    e = np.linalg.norm(p[:, [1, 2, 0]] - p, axis=2)
    area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
                        (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    return float(e.max()), float(math.sqrt(area.sum() / len(conn)))


def write_case(path, mesh, plane, out_dir):
    path.write_text(f"""mesh             = {mesh}
plane            = {plane}
E                = {E}
nu               = {NU}
thickness        = {T}
path             = distributed
partitioner      = metis
solver           = cg_jacobi
rtol             = 1e-12
max_iter         = 200000
output_dir       = {out_dir}
bc.dirichlet     = left_sym:ux=0 ; bottom_sym:uy=0
bc.traction      = kirsch_exact
kirsch.sigma_inf = {SIGMA_INF}
kirsch.a         = {A}
""")


def fit(h, e):
    slope, _ = np.polyfit(np.log(h), np.log(e), 1)
    return float(slope)


def run(args):
    out_root = Path(args.out)
    results = {"setup": {"sigma_inf": SIGMA_INF, "a": A, "L": 1.0, "E": E, "nu": NU,
                         "thickness": T, "rtol": 1e-12, "ranks": args.np,
                         "path": "distributed", "partitioner": "metis"},
               "levels": {}, "planes": {}}
    for level in args.levels:
        mesh = Path(args.meshes) / f"kirsch_L{level}.msh"
        t0 = time.time()
        h_max, h_eff = mesh_measures(mesh)
        results["levels"][level] = {"mesh": os.path.relpath(mesh, ROOT), "h_max": h_max,
                                    "h_eff": h_eff}
        print(f"L{level}: h_max {h_max:.4e}, h_eff {h_eff:.4e} ({time.time() - t0:.1f} s)")

    for plane in ("plane_stress", "plane_strain"):
        rows = {}
        for level in args.levels:
            out_dir = out_root / plane / f"L{level}"
            case = out_root / plane / f"L{level}.case"
            case.parent.mkdir(parents=True, exist_ok=True)
            write_case(case, results["levels"][level]["mesh"], plane, out_dir)
            if not args.skip_run:
                cmd = [args.mpiexec, "-n", str(args.np), args.solver, str(case)]
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    sys.exit(f"{' '.join(cmd)} failed:\n{proc.stdout}\n{proc.stderr}")
                print(f"{plane} L{level}: {time.time() - t0:.1f} s")
            tj = json.loads((out_dir / "timings.json").read_text())
            k = tj["kirsch"]
            rows[level] = {
                "dofs": tj["dofs"], "nnz": tj["nnz"],
                "iterations": tj["solver"]["iterations"],
                "true_rel_residual": tj["solver"]["true_rel_residual"],
                "converged": tj["solver"]["converged"],
                "solve_time_max": tj["timings"]["solve"]["max"],
                **{m: k[m] for m, _, _ in METRICS},
                "scf_numerical": k["scf"]["numerical"],
                "scf_analytical_at_centroid": k["scf"]["analytical_at_centroid"],
                "scf_centroid": [k["scf"]["x"], k["scf"]["y"]],
            }
            if not rows[level]["converged"]:
                sys.exit(f"{plane} L{level} did not converge")
        lv = list(args.levels)
        h = np.array([results["levels"][l]["h_eff"] for l in lv])
        fine = [i for i, l in enumerate(lv) if l >= 2]
        rates = {}
        for m, _, expected in METRICS:
            e = np.array([rows[l][m] for l in lv])
            rates[m] = {
                "expected": expected,
                "fit_all": fit(h, e),
                "fit_levels_2_plus": fit(h[fine], e[fine]) if len(fine) >= 2 else None,
                "consecutive": [float(math.log(e[i] / e[i + 1]) / math.log(h[i] / h[i + 1]))
                                for i in range(len(lv) - 1)],
            }
        dofs = np.array([rows[l]["dofs"] for l in lv], dtype=float)
        its = np.array([rows[l]["iterations"] for l in lv], dtype=float)
        rates["iterations_vs_dofs"] = {
            "expected": 0.5,
            "fit_all": float(np.polyfit(np.log(dofs), np.log(its), 1)[0]),
            "fit_levels_2_plus": float(np.polyfit(np.log(dofs[fine]), np.log(its[fine]), 1)[0]),
        }
        results["planes"][plane] = {"levels": rows, "rates": rates}
    return results


# ---------------------------------------------------------------------------- figures

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_axisbelow(True)
    if ax.get_xscale() == "log":  # 1-2-5 labels; default minor labels collide
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())


def figures(results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    lv = list(args.levels)
    ps = results["planes"]["plane_stress"]
    h = np.array([results["levels"][l]["h_eff"] for l in lv])
    written = []

    def save(fig, name):
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        path = fig_dir / name
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))

    # (a) errors vs h_eff
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for (m, label, _), c, mk in zip(METRICS, COLORS, MARKERS):
        e = np.array([ps["levels"][l][m] for l in lv])
        r = ps["rates"][m]
        ax.loglog(h, e, color=c, marker=mk, markersize=7, linewidth=2,
                  label=f"{label}: fit {r['fit_all']:.2f} (L2-5: {r['fit_levels_2_plus']:.2f})")
    ymin, ymax = ax.get_ylim()
    for slope, ls in ((1, "--"), (2, ":")):
        e0 = ps["levels"][lv[0]]["l2_rel_stress_error"] if slope == 1 else \
            ps["levels"][lv[0]]["l2_rel_displacement_error"]
        ref = e0 * 0.5 * (h / h[0]) ** slope
        ax.loglog(h, ref, color=MUTED, linestyle=ls, linewidth=1.2, label=f"reference slope {slope}")
    style(ax, r"$h_\mathrm{eff} = \sqrt{\mathrm{area}/n_\mathrm{tri}}$", "relative error",
          "Kirsch plate, plane stress: relative errors vs mesh size")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    save(fig, "convergence_errors.png")

    # (b) sigma_tt along the hole
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    th = np.linspace(0, 90, 361)
    ax.plot(th, 1 - 2 * np.cos(np.radians(2 * th)), color=INK, linewidth=1.2, zorder=3,
            label=r"analytical, $r = a$")
    for l, c, mk in zip((2, 5), COLORS, MARKERS):
        d = np.genfromtxt(Path(args.out) / "plane_stress" / f"L{l}" / "hole_sigma_tt.csv",
                          delimiter=",", names=True)
        ax.plot(np.degrees(d["theta"]), d["sigma_tt_num"] / SIGMA_INF, color=c, marker=mk,
                markersize=3.5 if l == 5 else 8, linestyle="none",
                markerfacecolor="none" if l == 5 else c, markeredgewidth=0.9,
                label=f"numerical L{l} (element centroids next to the hole)")
    style(ax, r"$\theta$ [deg]", r"$\sigma_{\theta\theta} / \sigma_\infty$",
          r"Hoop stress along the hole, plane stress")
    ax.set_xlim(0, 90)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    save(fig, "hole_sigma_tt.png")

    # (c) sigma_xx along x = 0
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.linspace(A, 1.0, 400)
    q2, q4 = (A / y) ** 2, (A / y) ** 4
    ax.plot(y / A, 0.5 * (1 + q2) + 0.5 * (1 + 3 * q4), color=INK, linewidth=1.2, zorder=3,
            label=r"analytical, $x = 0$")
    for l, c, mk in zip((2, 5), COLORS, MARKERS):
        d = np.genfromtxt(Path(args.out) / "plane_stress" / f"L{l}" / "section_x0_sigma_xx.csv",
                          delimiter=",", names=True)
        ax.plot(d["y"] / A, d["sigma_xx_num"] / SIGMA_INF, color=c, marker=mk,
                markersize=3.5 if l == 5 else 8, linestyle="none",
                markerfacecolor="none" if l == 5 else c, markeredgewidth=0.9,
                label=f"numerical L{l} (element centroids next to x = 0)")
    style(ax, r"$y / a$", r"$\sigma_{xx} / \sigma_\infty$",
          r"$\sigma_{xx}$ along the symmetry line $x = 0$, plane stress")
    ax.set_xlim(1, 10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    save(fig, "section_x0_sigma_xx.png")

    # (d) SCF vs h_eff
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    num = [ps["levels"][l]["scf_numerical"] for l in lv]
    ana = [ps["levels"][l]["scf_analytical_at_centroid"] for l in lv]
    ax.semilogx(h, num, color=COLORS[0], marker=MARKERS[0], markersize=7, linewidth=2,
                label=r"numerical $\sigma_{xx,h}/\sigma_\infty$ (element nearest $(0, a)$)")
    ax.semilogx(h, ana, color=COLORS[1], marker=MARKERS[1], markersize=7, linewidth=2,
                label=r"analytical $\sigma_{xx}/\sigma_\infty$ at the same centroid")
    ax.axhline(3.0, color=MUTED, linestyle="--", linewidth=1.2, label="theoretical SCF = 3")
    style(ax, r"$h_\mathrm{eff}$", "stress concentration factor",
          "SCF at (r = a, θ = 90°), plane stress")
    ax.invert_xaxis()
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    save(fig, "scf.png")

    # (e) PCG iterations vs DOFs
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    dofs = np.array([ps["levels"][l]["dofs"] for l in lv], dtype=float)
    its = np.array([ps["levels"][l]["iterations"] for l in lv], dtype=float)
    slope = ps["rates"]["iterations_vs_dofs"]["fit_all"]
    c0 = np.exp(np.polyfit(np.log(dofs), np.log(its), 1)[1])
    ax.loglog(dofs, its, color=COLORS[0], marker=MARKERS[0], markersize=7, linewidth=2,
              label="CG + Jacobi iterations (rtol 1e-12)")
    ax.loglog(dofs, c0 * dofs ** slope, color=MUTED, linestyle="--", linewidth=1.2,
              label=f"least-squares fit: slope {slope:.3f}")
    style(ax, "DOFs", "PCG iterations", "Jacobi-PCG iterations vs problem size, plane stress")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))  # one label per decade
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    save(fig, "pcg_iterations.png")
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--np", type=int, default=8, help="MPI ranks (default 8)")
    p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--solver", default=str(ROOT / "build" / "cstfem_solve"))
    p.add_argument("--mpiexec", default="mpiexec")
    p.add_argument("--meshes", default=str(ROOT / "meshes"))
    p.add_argument("--out", default=str(ROOT / "results" / "convergence"))
    p.add_argument("--figures", default=str(ROOT / "docs" / "figures"))
    p.add_argument("--json", default=str(ROOT / "docs" / "verification_results.json"))
    p.add_argument("--skip-run", action="store_true", help="reuse existing outputs")
    args = p.parse_args(argv)
    os.chdir(ROOT)

    results = run(args)
    results["figures"] = figures(results, args)
    Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.json}")

    for plane, d in results["planes"].items():
        print(f"\n{plane}")
        print("| L | DOFs | h_max | h_eff | iter | true res | disp L2 | energy | stress L2 | "
              "SCF num | SCF ana @centroid |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for l in args.levels:
            r, m = d["levels"][l], results["levels"][l]
            print(f"| {l} | {r['dofs']} | {m['h_max']:.4e} | {m['h_eff']:.4e} | {r['iterations']} | "
                  f"{r['true_rel_residual']:.2e} | {r['l2_rel_displacement_error']:.4e} | "
                  f"{r['energy_rel_error']:.4e} | {r['l2_rel_stress_error']:.4e} | "
                  f"{r['scf_numerical']:.4f} | {r['scf_analytical_at_centroid']:.4f} |")
        for m, label, _ in METRICS:
            rr = d["rates"][m]
            print(f"  {label}: expected {rr['expected']}, fit all {rr['fit_all']:.3f}, "
                  f"fit L2+ {rr['fit_levels_2_plus']:.3f}, consecutive "
                  + ", ".join(f"{x:.3f}" for x in rr["consecutive"]))
        it = d["rates"]["iterations_vs_dofs"]
        print(f"  iterations ~ DOFs^p: p = {it['fit_all']:.3f} (L2+: {it['fit_levels_2_plus']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
