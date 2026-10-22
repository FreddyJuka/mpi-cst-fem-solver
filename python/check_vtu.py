#!/usr/bin/env python3
"""Independent check of cstfem_solve output (xml + base64 + numpy; no VTK, no meshio).

Reads <output_dir>/solution.pvtu and all its pieces, decodes every DataArray, checks point
and cell counts and that no array contains NaN/Inf, and merges the pieces into one global
view: nodes deduplicated with the `owned` flag and keyed by `gmsh_tag`, cells keyed by
`element_id` (each must appear exactly once). If hole_sigma_tt.csv exists, its numerical
sigma_tt is compared with the cell stresses.

With --compare REF_DIR, the displacement (by Gmsh tag), all common cell arrays (by element
id) and the CSV files are compared with another output directory (e.g. the serial run).
Exits non-zero if any check fails.
"""
import argparse
import base64
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DTYPES = {"Float64": "f8", "Float32": "f4", "Int64": "i8", "Int32": "i4", "UInt8": "u1"}
HEADER_B64 = 12  # base64 length of one UInt64 header (8 bytes)
CSV_FILES = ("hole_sigma_tt.csv", "section_x0_sigma_xx.csv")


def decode(da, byte_order):
    if da.get("format") != "binary":
        raise ValueError(f"DataArray {da.get('Name')}: only format=binary is supported")
    endian = "<" if byte_order == "LittleEndian" else ">"
    text = "".join(da.text.split())
    # VTK encodes the UInt64 byte-count header and the payload as separate base64 blocks.
    nbytes = int(np.frombuffer(base64.b64decode(text[:HEADER_B64]), endian + "u8")[0])
    payload = base64.b64decode(text[HEADER_B64:])
    if len(payload) != nbytes:
        raise ValueError(f"DataArray {da.get('Name')}: header says {nbytes} bytes, "
                         f"payload has {len(payload)}")
    arr = np.frombuffer(payload, endian + DTYPES[da.get("type")])
    ncomp = int(da.get("NumberOfComponents", "1"))
    return arr.reshape(-1, ncomp) if ncomp > 1 else arr


def read_piece(path):
    root = ET.parse(path).getroot()
    if root.get("type") != "UnstructuredGrid" or root.get("header_type") != "UInt64":
        raise ValueError(f"{path}: expected UnstructuredGrid with header_type UInt64")
    order = root.get("byte_order")
    piece = root.find("UnstructuredGrid/Piece")
    return {
        "npoints": int(piece.get("NumberOfPoints")),
        "ncells": int(piece.get("NumberOfCells")),
        "points": decode(piece.find("Points/DataArray"), order),
        "point": {d.get("Name"): decode(d, order) for d in piece.findall("PointData/DataArray")},
        "cell": {d.get("Name"): decode(d, order) for d in piece.findall("CellData/DataArray")},
        "cells": {d.get("Name"): decode(d, order) for d in piece.findall("Cells/DataArray")},
    }


def load(out_dir, failures):
    """Checks every piece and returns the merged global view."""
    master = ET.parse(out_dir / "solution.pvtu").getroot()
    if master.get("type") != "PUnstructuredGrid":
        failures.append("solution.pvtu is not a PUnstructuredGrid")
    pgrid = master.find("PUnstructuredGrid")
    sources = [pc.get("Source") for pc in pgrid.findall("Piece")]
    p_point = [d.get("Name") for d in pgrid.findall("PPointData/PDataArray")]
    p_cell = [d.get("Name") for d in pgrid.findall("PCellData/PDataArray")]
    print(f"{out_dir}: pvtu lists {len(sources)} piece(s)")

    nodes = {}  # gmsh_tag -> (x, y, displacement), owned copies only
    cells = {}  # element_id -> {array: value}
    centroids = {}
    for src in sources:
        pc = read_piece(out_dir / src)
        np_, nc = pc["npoints"], pc["ncells"]
        if pc["points"].shape != (np_, 3):
            failures.append(f"{src}: points shape {pc['points'].shape}")
        conn, offs, types = (pc["cells"][k] for k in ("connectivity", "offsets", "types"))
        if conn.size != 3 * nc or offs.size != nc or types.size != nc:
            failures.append(f"{src}: cell array sizes do not match NumberOfCells")
        if not np.all(types == 5) or not np.array_equal(offs, 3 * np.arange(1, nc + 1)):
            failures.append(f"{src}: cells are not all linear triangles")
        if nc and (conn.min() < 0 or conn.max() >= np_):
            failures.append(f"{src}: connectivity out of range")
        for kind, arrays, count in (("point", pc["point"], np_), ("cell", pc["cell"], nc)):
            for name, a in arrays.items():
                if a.shape[0] != count:
                    failures.append(f"{src}: {kind} array {name} has {a.shape[0]} tuples")
                if not np.all(np.isfinite(a)):
                    failures.append(f"{src}: {kind} array {name} contains NaN/Inf")
        if list(pc["point"]) != p_point or list(pc["cell"]) != p_cell:
            failures.append(f"{src}: arrays differ from the .pvtu declaration")

        owned = pc["point"].get("owned", np.ones(np_, dtype=int))
        tags = pc["point"].get("gmsh_tag", np.arange(np_))
        for i in np.flatnonzero(owned):
            t = int(tags[i])
            if t in nodes:
                failures.append(f"node tag {t} owned by more than one piece")
            nodes[t] = (pc["points"][i, :2], pc["point"]["displacement"][i])
        ids = pc["cell"].get("element_id", np.arange(nc))
        cent = pc["points"][conn.reshape(-1, 3)][:, :, :2].mean(axis=1) if nc else np.zeros((0, 2))
        for k in range(nc):
            e = int(ids[k])
            if e in cells:
                failures.append(f"element {e} appears in more than one piece")
            cells[e] = {name: a[k] for name, a in pc["cell"].items()}
            centroids[e] = cent[k]
        print(f"  {src}: {np_} points ({int(owned.sum())} owned), {nc} cells")
    print(f"  merged: {len(nodes)} owned nodes, {len(cells)} cells; "
          f"cell arrays {p_cell}")
    return nodes, cells, centroids


def check_hole_csv(out_dir, cells, centroids, tol, failures):
    csv = out_dir / "hole_sigma_tt.csv"
    if not csv.exists():
        return
    rows = np.atleast_1d(np.genfromtxt(csv, delimiter=",", names=True))
    ids = np.array(list(centroids))
    cent = np.array([centroids[e] for e in ids])
    worst, scale = 0.0, np.abs(rows["sigma_tt_num"]).max()
    for row in rows:
        d = np.hypot(cent[:, 0] - row["x"], cent[:, 1] - row["y"])
        k = int(np.argmin(d))
        if d[k] > 1e-12:
            failures.append(f"CSV centroid ({row['x']}, {row['y']}) not found in VTU")
            continue
        c = cells[int(ids[k])]
        th = np.arctan2(cent[k, 1], cent[k, 0])
        cs, sn = np.cos(th), np.sin(th)
        stt = c["sigma_xx"] * sn * sn + c["sigma_yy"] * cs * cs - 2 * c["tau_xy"] * sn * cs
        worst = max(worst, abs(stt - row["sigma_tt_num"]) / scale)
    print(f"  hole sigma_tt: {rows.size} CSV rows vs VTU, max rel. difference {worst:.2e}")
    if worst > tol:
        failures.append(f"hole sigma_tt differs from the CSV by {worst:.2e} > {tol}")


def rel_diff(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = np.abs(b).max()
    return float(np.abs(a - b).max() / scale) if scale > 0 else float(np.abs(a - b).max())


def compare(out_dir, ref_dir, view, ref, tol, failures):
    (nodes, cells, _), (rnodes, rcells, _) = view, ref
    print(f"compare {out_dir} with {ref_dir}:")
    if set(nodes) != set(rnodes) or set(cells) != set(rcells):
        failures.append("node tags or element ids differ from the reference")
        return
    tags = sorted(rnodes)
    d = rel_diff([nodes[t][1] for t in tags], [rnodes[t][1] for t in tags])
    print(f"  displacement (by Gmsh tag): max rel. difference {d:.2e}")
    if d > tol:
        failures.append(f"displacement differs by {d:.2e} > {tol}")
    ids = sorted(rcells)
    skip = {"partition_id", "element_id"}
    for name in rcells[ids[0]]:
        if name in skip or name not in cells[ids[0]]:
            continue
        d = rel_diff([cells[e][name] for e in ids], [rcells[e][name] for e in ids])
        print(f"  cell {name:<15} (by element id): max rel. difference {d:.2e}")
        if d > tol:
            failures.append(f"cell array {name} differs by {d:.2e} > {tol}")
    for f in CSV_FILES:
        if not (ref_dir / f).exists():
            continue
        a = np.genfromtxt(out_dir / f, delimiter=",", skip_header=1, ndmin=2)
        b = np.genfromtxt(ref_dir / f, delimiter=",", skip_header=1, ndmin=2)
        if a.shape != b.shape:
            failures.append(f"{f}: shape {a.shape} vs reference {b.shape}")
            continue
        d = max(rel_diff(a[:, j], b[:, j]) for j in range(b.shape[1]))
        print(f"  {f}: {b.shape[0]} rows, max rel. difference per column {d:.2e}")
        if d > tol:
            failures.append(f"{f} differs by {d:.2e} > {tol}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("output_dir")
    p.add_argument("--compare", metavar="REF_DIR", help="reference output directory")
    p.add_argument("--tol", type=float, default=1e-8,
                   help="relative tolerance for --compare (default 1e-8)")
    p.add_argument("--csv-tol", type=float, default=1e-9,
                   help="relative tolerance VTU vs hole CSV (default 1e-9)")
    args = p.parse_args(argv)
    out_dir = Path(args.output_dir)
    failures = []

    view = load(out_dir, failures)
    check_hole_csv(out_dir, view[1], view[2], args.csv_tol, failures)
    if args.compare:
        ref_dir = Path(args.compare)
        ref = load(ref_dir, failures)
        compare(out_dir, ref_dir, view, ref, args.tol, failures)

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
