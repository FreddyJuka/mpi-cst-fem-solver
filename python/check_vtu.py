#!/usr/bin/env python3
"""Independent check of cstfem_solve output (xml + base64 + numpy; no VTK, no meshio).

Reads <output_dir>/solution.pvtu and its pieces, decodes every DataArray, checks point and
cell counts, checks that no array contains NaN/Inf, and (if hole_sigma_tt.csv exists)
compares the numerical hole sigma_tt in the CSV with the cell stresses in the VTU.
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
    out = {
        "npoints": int(piece.get("NumberOfPoints")),
        "ncells": int(piece.get("NumberOfCells")),
        "points": decode(piece.find("Points/DataArray"), order),
        "point": {d.get("Name"): decode(d, order) for d in piece.findall("PointData/DataArray")},
        "cell": {d.get("Name"): decode(d, order) for d in piece.findall("CellData/DataArray")},
    }
    out["cells"] = {d.get("Name"): decode(d, order) for d in piece.findall("Cells/DataArray")}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("output_dir")
    p.add_argument("--tol", type=float, default=1e-9, help="relative tolerance vs the CSV")
    args = p.parse_args(argv)
    out_dir = Path(args.output_dir)
    failures = []

    master = ET.parse(out_dir / "solution.pvtu").getroot()
    if master.get("type") != "PUnstructuredGrid":
        failures.append("solution.pvtu is not a PUnstructuredGrid")
    pgrid = master.find("PUnstructuredGrid")
    sources = [pc.get("Source") for pc in pgrid.findall("Piece")]
    p_point = [d.get("Name") for d in pgrid.findall("PPointData/PDataArray")]
    p_cell = [d.get("Name") for d in pgrid.findall("PCellData/PDataArray")]
    print(f"pvtu: {len(sources)} piece(s): {', '.join(sources)}")

    total_cells = 0
    pieces = []
    for src in sources:
        pc = read_piece(out_dir / src)
        pieces.append(pc)
        np_, nc = pc["npoints"], pc["ncells"]
        total_cells += nc
        print(f"{src}: {np_} points, {nc} cells; point arrays {list(pc['point'])}; "
              f"cell arrays {list(pc['cell'])}")
        if pc["points"].shape != (np_, 3):
            failures.append(f"{src}: points shape {pc['points'].shape}")
        conn, offs, types = (pc["cells"][k] for k in ("connectivity", "offsets", "types"))
        if conn.size != 3 * nc or offs.size != nc or types.size != nc:
            failures.append(f"{src}: cell array sizes do not match NumberOfCells")
        if not np.all(types == 5) or not np.array_equal(offs, 3 * np.arange(1, nc + 1)):
            failures.append(f"{src}: cells are not all linear triangles")
        if conn.min() < 0 or conn.max() >= np_:
            failures.append(f"{src}: connectivity out of range")
        for kind, arrays, count in (("point", pc["point"], np_), ("cell", pc["cell"], nc)):
            for name, a in arrays.items():
                if a.shape[0] != count:
                    failures.append(f"{src}: {kind} array {name} has {a.shape[0]} tuples")
                if not np.all(np.isfinite(a)):
                    failures.append(f"{src}: {kind} array {name} contains NaN/Inf")
        if list(pc["point"]) != p_point or list(pc["cell"]) != p_cell:
            failures.append(f"{src}: arrays differ from the .pvtu declaration")

    csv = out_dir / "hole_sigma_tt.csv"
    if csv.exists():
        rows = np.genfromtxt(csv, delimiter=",", names=True)
        pc = pieces[0]  # serial output: one piece
        pts, conn = pc["points"], pc["cells"]["connectivity"].reshape(-1, 3)
        cent = pts[conn][:, :, :2].mean(axis=1)
        sxx, syy, txy = (pc["cell"][k] for k in ("sigma_xx", "sigma_yy", "tau_xy"))
        worst, scale = 0.0, np.abs(rows["sigma_tt_num"]).max()
        for row in rows:
            d = np.hypot(cent[:, 0] - row["x"], cent[:, 1] - row["y"])
            e = int(np.argmin(d))
            if d[e] > 1e-12:
                failures.append(f"CSV centroid ({row['x']}, {row['y']}) not found in VTU")
                continue
            th = np.arctan2(cent[e, 1], cent[e, 0])
            c, s = np.cos(th), np.sin(th)
            stt = sxx[e] * s * s + syy[e] * c * c - 2 * txy[e] * s * c
            worst = max(worst, abs(stt - row["sigma_tt_num"]) / scale)
        print(f"hole sigma_tt: {rows.size} CSV rows vs VTU, max rel. difference {worst:.2e}")
        if worst > args.tol:
            failures.append(f"hole sigma_tt differs from the CSV by {worst:.2e} > {args.tol}")

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
