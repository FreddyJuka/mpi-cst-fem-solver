#!/usr/bin/env python3
"""Gmsh generator for the quarter-plate Kirsch problem (SPEC sections 8 and 10).

Geometry: [0, L] x [0, L] minus a disk of radius a centered at the origin.
Physical groups: hole, left_sym, bottom_sym, right, top, domain.
Output: Gmsh .msh format 4.1, ASCII, linear triangles only.
"""
import argparse
import sys
from pathlib import Path

import gmsh


def _classify_curves(a, L, tol):
    """Map physical-group name -> list of curve tags, using bounding boxes only."""
    groups = {"hole": [], "left_sym": [], "bottom_sym": [], "right": [], "top": []}
    for dim, tag in gmsh.model.getEntities(1):
        x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(dim, tag)
        if abs(x0) < tol and abs(x1) < tol:
            groups["left_sym"].append(tag)
        elif abs(y0) < tol and abs(y1) < tol:
            groups["bottom_sym"].append(tag)
        elif abs(x0 - L) < tol and abs(x1 - L) < tol:
            groups["right"].append(tag)
        elif abs(y0 - L) < tol and abs(y1 - L) < tol:
            groups["top"].append(tag)
        elif max(x1, y1) <= a + tol and min(x0, y0) > -tol:
            groups["hole"].append(tag)
        else:
            raise RuntimeError(f"unclassified boundary curve {tag}: bbox {(x0, y0, x1, y1)}")
    for name, tags in groups.items():
        if not tags:
            raise RuntimeError(f"no curve found for physical group '{name}'")
    return groups


def generate(a=0.1, L=1.0, h_hole=0.02, h_far=0.1, level=0, output="kirsch.msh", verbose=False):
    """Generate the Kirsch quarter-plate mesh and write it to `output` (.msh v4.1 ASCII).

    Sizes h_hole and h_far are scaled by 2**(-level). Returns the output path.
    """
    if not 0 < a < L:
        raise ValueError("require 0 < a < L")
    scale = 2.0 ** (-level)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("kirsch")
        occ = gmsh.model.occ
        plate = occ.addRectangle(0, 0, 0, L, L)
        disk = occ.addDisk(0, 0, 0, a, a)
        cut, _ = occ.cut([(2, plate)], [(2, disk)])
        occ.synchronize()
        surfaces = [t for d, t in cut if d == 2]
        if len(surfaces) != 1:
            raise RuntimeError(f"expected one surface after cut, got {len(surfaces)}")

        # The disk is a full circle; the cut keeps only the part inside the plate (a quarter
        # arc). Identify curves geometrically.
        # getBoundingBox pads boxes by ~1e-7, so classify with a looser tolerance.
        tol = 1e-6 * L
        groups = _classify_curves(a, L, tol)
        for name, tags in groups.items():
            gmsh.model.addPhysicalGroup(1, tags, name=name)
        gmsh.model.addPhysicalGroup(2, surfaces, name="domain")

        # Size field: Distance from the hole + Threshold.
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "CurvesList", groups["hole"])
        gmsh.model.mesh.field.setNumber(dist, "Sampling", 200)
        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", h_hole * scale)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", h_far * scale)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", L / 2)
        gmsh.model.mesh.field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Linear triangles only, Frontal-Delaunay.
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)

        gmsh.model.mesh.generate(2)
        gmsh.write(str(output))
    finally:
        gmsh.finalize()
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", type=float, default=0.1, help="hole radius (default 0.1)")
    p.add_argument("--L", type=float, default=1.0, help="plate side (default 1.0)")
    p.add_argument("--h-hole", type=float, default=0.02, help="size at the hole, level 0")
    p.add_argument("--h-far", type=float, default=0.1, help="size far from the hole, level 0")
    p.add_argument("--level", type=int, default=0, help="refinement level; h scales by 2^-level")
    p.add_argument("-o", "--output", default="meshes/kirsch_L0.msh")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    out = generate(args.a, args.L, args.h_hole, args.h_far, args.level, args.output, args.verbose)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
