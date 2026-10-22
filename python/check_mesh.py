#!/usr/bin/env python3
"""Independent checker for Gmsh .msh v4.1 ASCII meshes (does not use the gmsh API).

Doubles as a prototype of the C++ reader logic. Exits non-zero if any check fails.
"""
import argparse
import math
import sys
from collections import defaultdict


def parse_msh(path):
    with open(path) as f:
        tokens_by_section = {}
        name, buf = None, []
        for line in f:
            line = line.strip()
            if line.startswith("$End"):
                tokens_by_section[name] = buf
                name, buf = None, []
            elif line.startswith("$"):
                name, buf = line[1:], []
            elif name is not None:
                buf.append(line)
    for req in ("MeshFormat", "PhysicalNames", "Entities", "Nodes", "Elements"):
        if req not in tokens_by_section:
            raise ValueError(f"missing section ${req}")

    header = tokens_by_section["MeshFormat"][0].split()

    phys = {}  # (dim, tag) -> name
    lines = tokens_by_section["PhysicalNames"]
    for ln in lines[1:int(lines[0]) + 1]:
        dim, tag, rest = ln.split(None, 2)
        phys[(int(dim), int(tag))] = rest.strip().strip('"')

    # Entities: dim,tag -> list of physical tags
    ent_lines = tokens_by_section["Entities"]
    nP, nC, nS, nV = map(int, ent_lines[0].split())
    entity_phys = {}
    idx = 1
    for dim, count in ((0, nP), (1, nC), (2, nS), (3, nV)):
        for _ in range(count):
            t = ent_lines[idx].split()
            idx += 1
            tag = int(t[0])
            if dim == 0:
                nphys_pos = 4
            else:
                nphys_pos = 7
            nphys = int(t[nphys_pos])
            entity_phys[(dim, tag)] = [int(x) for x in t[nphys_pos + 1:nphys_pos + 1 + nphys]]

    # Nodes (ASCII v4.1): numBlocks numNodes minTag maxTag; per block: dim tag param n; tags; coords
    nl = tokens_by_section["Nodes"]
    nblocks = int(nl[0].split()[0])
    coords = {}
    node_entity = {}  # node tag -> (dim, entity tag)
    i = 1
    for _ in range(nblocks):
        edim, etag, _param, n = map(int, nl[i].split())
        i += 1
        tags = [int(nl[i + k]) for k in range(n)]
        i += n
        for k in range(n):
            x, y, z = map(float, nl[i + k].split())
            coords[tags[k]] = (x, y, z)
            node_entity[tags[k]] = (edim, etag)
        i += n

    # Elements: numBlocks ...; per block: dim tag type n; lines "elemTag n1 n2 ..."
    el = tokens_by_section["Elements"]
    nblocks = int(el[0].split()[0])
    tris, segs = [], defaultdict(list)  # segs: group name -> list of (n1, n2)
    i = 1
    for _ in range(nblocks):
        edim, etag, etype, n = map(int, el[i].split())
        i += 1
        ptags = entity_phys.get((edim, etag), [])
        names = [phys[(edim, p)] for p in ptags if (edim, p) in phys]
        for k in range(n):
            t = el[i + k].split()
            conn = tuple(int(x) for x in t[1:])
            if etype == 2:
                tris.append((conn, names))
            elif etype == 1:
                for nm in names:
                    segs[nm].append(conn)
            else:
                raise ValueError(f"unsupported element type {etype}")
        i += n
    return header, coords, tris, segs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mesh")
    p.add_argument("--a", type=float, default=0.1)
    p.add_argument("--L", type=float, default=1.0)
    args = p.parse_args(argv)
    a, L = args.a, args.L
    tol = 1e-9 * L
    failures = []

    header, coords, tris, segs = parse_msh(args.mesh)
    if header[:3] != ["4.1", "0", "8"]:
        failures.append(f"header is {' '.join(header)}, expected '4.1 0 8'")
    print(f"header: {' '.join(header)}")

    used_nodes = {n for conn, _ in tris for n in conn}
    print(f"nodes: {len(coords)} (used by triangles: {len(used_nodes)})")
    print(f"triangles: {len(tris)}")
    for nm in sorted(segs):
        print(f"lines [{nm}]: {len(segs[nm])}")
    for req in ("hole", "left_sym", "bottom_sym", "right", "top"):
        if req not in segs:
            failures.append(f"no line elements in physical group '{req}'")

    neg = 0
    qmin, qmax = math.inf, -math.inf
    emin, emax = math.inf, -math.inf
    for conn, _ in tris:
        (x1, y1, _), (x2, y2, _), (x3, y3, _) = (coords[n] for n in conn)
        area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if area2 <= 0:
            neg += 1
        e = [math.hypot(x2 - x1, y2 - y1), math.hypot(x3 - x2, y3 - y2),
             math.hypot(x1 - x3, y1 - y3)]
        emin, emax = min(emin, *e), max(emax, *e)
        # Radius ratio 2 r_in / R_circ = 16 A^2 / (abc (a+b+c)), in (0, 1]; 1 = equilateral.
        A = abs(area2) / 2
        q = 16 * A * A / (e[0] * e[1] * e[2] * sum(e)) if A > 0 else 0.0
        qmin, qmax = min(qmin, q), max(qmax, q)
    print(f"negative/zero-area triangles: {neg}")
    print(f"edge length: min {emin:.6g}, max {emax:.6g}")
    print(f"quality (radius ratio): min {qmin:.4f}, max {qmax:.4f}")
    if neg:
        failures.append(f"{neg} triangles with non-positive signed area")

    def check_group(name, dist):
        nodes = {n for conn in segs.get(name, []) for n in conn}
        bad = sum(1 for n in nodes if dist(coords[n]) >= tol)
        print(f"group {name}: {len(nodes)} nodes, {bad} off geometry")
        if bad:
            failures.append(f"{bad} '{name}' nodes violate tolerance {tol:g}")

    check_group("hole", lambda c: abs(math.hypot(c[0], c[1]) - a))
    check_group("left_sym", lambda c: abs(c[0]))
    check_group("bottom_sym", lambda c: abs(c[1]))
    check_group("right", lambda c: abs(c[0] - L))
    check_group("top", lambda c: abs(c[1] - L))

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
