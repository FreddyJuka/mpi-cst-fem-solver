#!/usr/bin/env python3
"""Hero figure docs/figures/kirsch_fields.png: partition and sigma_xx of the Kirsch plate.

Runs cases/kirsch_L3.case with 4 ranks (distributed path) into results/kirsch_fields/ unless
that directory already has a solution, decodes the .vtu pieces with check_vtu.read_piece and
plots two panels with tripcolor: (a) partition_id per element with mesh edges, for the whole
quarter plate (so that all ranks are visible; the zoom window of (b) is dashed), (b) sigma_xx /
sigma_inf per element, zoomed on the hole region.
"""
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_vtu import read_piece  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "kirsch_fields"
CASE = ROOT / "cases" / "kirsch_L3.case"
FIG = ROOT / "docs" / "figures" / "kirsch_fields.png"
EXE = ROOT / "build" / "cstfem_solve"
NP = 4
ZOOM = 0.35  # zoom window [0, ZOOM]^2 of panel (b); hole radius a = 0.1, plate side L = 1


def run():
    if (OUT / "solution.pvtu").exists():
        return
    text = re.sub(r"^output_dir\s*=.*$", f"output_dir       = {OUT.relative_to(ROOT)}",
                  CASE.read_text(), flags=re.M)
    if not re.search(r"^path\s*=", text, flags=re.M):
        text += "\npath             = distributed\n"
    case = ROOT / "results" / "kirsch_fields.case"
    case.write_text(text)
    subprocess.run(["mpiexec", "--oversubscribe", "-n", str(NP), str(EXE), str(case.relative_to(ROOT))],
                   cwd=ROOT, check=True)


def sigma_inf():
    m = re.search(r"^kirsch\.sigma_inf\s*=\s*(\S+)", CASE.read_text(), flags=re.M)
    return float(m.group(1))


def load():
    master = ET.parse(OUT / "solution.pvtu").getroot()
    sources = [p.get("Source") for p in master.find("PUnstructuredGrid").findall("Piece")]
    xy, tris, pid, sxx = [], [], [], []
    base = 0
    for src in sources:
        pc = read_piece(OUT / src)
        xy.append(pc["points"][:, :2])
        tris.append(pc["cells"]["connectivity"].reshape(-1, 3) + base)
        pid.append(pc["cell"]["partition_id"])
        sxx.append(pc["cell"]["sigma_xx"])
        base += pc["npoints"]
    return (np.vstack(xy), np.vstack(tris), np.concatenate(pid), np.concatenate(sxx))


def main():
    run()
    xy, tris, pid, sxx = load()
    s = sigma_inf()
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    cmap_part = plt.get_cmap("tab10", NP)
    ax = axes[0]
    ax.tripcolor(tri, facecolors=pid, cmap=cmap_part, vmin=-0.5, vmax=NP - 0.5,
                 edgecolors="k", linewidth=0.08)
    ax.plot([0, ZOOM, ZOOM, 0, 0], [0, 0, ZOOM, ZOOM, 0], "w--", linewidth=1.2)
    ax.set_title(f"(a) partition_id per element ({NP} ranks, METIS)")
    handles = [plt.Line2D([], [], marker="s", linestyle="", color=cmap_part(i), label=f"rank {i}")
               for i in range(NP)]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    ax = axes[1]
    pc = ax.tripcolor(tri, facecolors=sxx / s, cmap="viridis")
    cb = fig.colorbar(pc, ax=ax, shrink=0.85)
    cb.set_label(r"$\sigma_{xx}/\sigma_\infty$")
    ax.set_title(r"(b) $\sigma_{xx}/\sigma_\infty$ per element")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[1].set_xlim(0, ZOOM)
    axes[1].set_ylim(0, ZOOM)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.suptitle(f"Kirsch plate with a hole, quarter model (kirsch_L3, {tris.shape[0]} elements)", fontsize=10)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}; sigma_xx/sigma_inf range {sxx.min() / s:.3f} .. {sxx.max() / s:.3f}")


if __name__ == "__main__":
    main()
