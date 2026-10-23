#!/usr/bin/env python3
"""Strong and weak scaling study driver (Milestone 6b).

Subcommands (all resumable; a run whose timings.json exists is skipped):
  meshes     generate the weak-scaling meshes and the L6 strong mesh; write cases/scaling_*.case
  probe      time the largest strong case (L6, np=1) and the largest weak case (200k, p=10)
  estimate   estimate the total study time from the probes and the L4/L5 warm-up shape
  run        run every group (warm-up + interleaved repetitions) and the triad benchmark
  aggregate  write docs/benchmark_results.json from the raw outputs
  parmetis   run the METIS vs ParMETIS comparison (needs a build with -DUSE_PARMETIS=ON)

Every run: path = distributed, partitioner as given, instrument = true, plane_stress,
rtol = 1e-10, mpiexec --bind-to core --map-by core. Raw outputs go to results/scaling/.
"""
import argparse
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

NP_LIST = [1, 2, 3, 4, 6, 8, 10]
STRONG_LEVELS = [4, 5, 6]
WEAK_DOFS_PER_RANK = {"50k": 50_000, "200k": 200_000}
REPS = 5
# Reduced plan (runtime budget): L6 strong and the 200k weak series use fewer np and repetitions.
REDUCED_NP = [1, 2, 4, 8]
REDUCED_REPS = 3
REDUCED_GROUPS = ("strong_L6", "weak_200k")


def group_np(name):
    return REDUCED_NP if name in REDUCED_GROUPS else NP_LIST


def group_reps(name):
    return REDUCED_REPS if name in REDUCED_GROUPS else REPS
PM_NP = [4, 8, 10]
SCALING = ROOT / "results" / "scaling"
CASES = ROOT / "cases"
MESHES = ROOT / "meshes"
EXE = ROOT / "build" / "cstfem_solve"
EXE_PM = ROOT / "build-parmetis" / "cstfem_solve"
TRIAD = ROOT / "build" / "benchmarks" / "stream_triad"
MPI_PREFIX = ["mpiexec", "--bind-to", "core", "--map-by", "core", "--report-bindings"]

CASE_TEMPLATE = """# Scaling study case ({title}); settings as in cases/kirsch_L*.case plus path/instrument.
# Paths are relative to the working directory (run from the repository root).
mesh             = {mesh}
plane            = plane_stress
E                = 210e9
nu               = 0.3
thickness        = 0.01
partitioner      = {partitioner}
path             = distributed
instrument       = true
solver           = cg_jacobi
rtol             = 1e-10
max_iter         = 20000
output_dir       = results/scaling/out
# boundary conditions
bc.dirichlet     = left_sym:ux=0 ; bottom_sym:uy=0
bc.traction      = kirsch_exact
kirsch.sigma_inf = 100e6
kirsch.a         = 0.1
"""


# ----------------------------------------------------------------------------- groups
def case_path(name):
    return CASES / f"scaling_{name}.case"


def groups():
    """name -> list of entries {key, np, case, exe, partitioner}."""
    g = {}
    for lvl in STRONG_LEVELS:
        g[f"strong_L{lvl}"] = [dict(key=f"np{p}", np=p, case=case_path(f"strong_L{lvl}"),
                                    exe=EXE, partitioner="metis") for p in group_np(f"strong_L{lvl}")]
    for tag in WEAK_DOFS_PER_RANK:
        g[f"weak_{tag}"] = [dict(key=f"np{p}", np=p, case=case_path(f"weak_{tag}_p{p}"),
                                 exe=EXE, partitioner="metis") for p in group_np(f"weak_{tag}")]
    g["pm_L5"] = [dict(key=f"{part}_np{p}", np=p, case=case_path(f"pm_L5_{part}"), exe=EXE_PM,
                       partitioner=part) for part in ("metis", "parmetis") for p in PM_NP]
    return g


def rotated(entries, r):
    k = r % len(entries)
    return entries[k:] + entries[:k]


# ----------------------------------------------------------------------------- meshes
def cmd_meshes(args):
    import mesh_kirsch
    SCALING.mkdir(parents=True, exist_ok=True)
    meta_path = SCALING / "meshes.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def case_for(name, mesh, partitioner="metis"):
        case_path(name).write_text(CASE_TEMPLATE.format(
            title=name, mesh=mesh.relative_to(ROOT), partitioner=partitioner))

    for lvl in STRONG_LEVELS:
        mesh = MESHES / f"kirsch_L{lvl}.msh"
        if not mesh.exists() or f"strong_L{lvl}" not in meta:
            t0 = time.time()
            if not mesh.exists():
                mesh_kirsch.generate(level=lvl, output=mesh)
            meta[f"strong_L{lvl}"] = dict(mesh=str(mesh.relative_to(ROOT)),
                                          dofs=2 * mesh_kirsch.count_nodes(mesh),
                                          bytes=mesh.stat().st_size, gen_seconds=time.time() - t0)
        case_for(f"strong_L{lvl}", mesh)
    case_for("pm_L5_metis", MESHES / "kirsch_L5.msh", "metis")
    case_for("pm_L5_parmetis", MESHES / "kirsch_L5.msh", "parmetis")

    for tag, per_rank in WEAK_DOFS_PER_RANK.items():
        for p in group_np(f"weak_{tag}"):
            name = f"weak_{tag}_p{p}"
            mesh = MESHES / f"scaling_{name}.msh"
            if not mesh.exists() or name not in meta:
                target = per_rank * p
                print(f"{name}: target {target} DOFs", flush=True)
                scale, dofs, tries, secs = mesh_kirsch.generate_target_dofs(
                    target, output=mesh, verbose=False)
                meta[name] = dict(mesh=str(mesh.relative_to(ROOT)), target_dofs=target, dofs=dofs,
                                  scale=scale, tries=tries, bytes=mesh.stat().st_size,
                                  gen_seconds=secs, dofs_per_rank=dofs / p)
                meta_path.write_text(json.dumps(meta, indent=1))
                print(f"  -> {dofs} DOFs ({dofs / p:.0f} per rank), {tries} meshes, {secs:.0f} s",
                      flush=True)
            case_for(name, mesh)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "mesh"} for k, v in meta.items()},
                     indent=1))


# ----------------------------------------------------------------------------- running
def run_dir(group, key, tag):
    return SCALING / group / key / tag


def run_one(group, e, tag, force_record_bindings=True):
    """One mpiexec run; returns (wall seconds, ran). Skips when timings.json exists."""
    d = run_dir(group, e["key"], tag)
    if (d / "timings.json").exists():
        return 0.0, False
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    out = d / "out"
    text = e["case"].read_text()
    text = re.sub(r"^output_dir\s*=.*$", f"output_dir       = {out.relative_to(ROOT)}", text,
                  flags=re.M)
    run_case = d / "run.case"
    run_case.write_text(text)
    cmd = MPI_PREFIX + ["-n", str(e["np"]), str(e["exe"]), str(run_case.relative_to(ROOT))]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    (d / "stdout.txt").write_text(r.stdout)
    (d / "stderr.txt").write_text(r.stderr)
    if r.returncode != 0 or not (out / "timings.json").exists():
        sys.exit(f"run failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    t = json.loads((out / "timings.json").read_text())
    if not t["solver"]["converged"]:
        sys.exit(f"not converged: {d}")
    shutil.copy(out / "timings.json", d / "timings.json")
    shutil.rmtree(out)
    (d / "wall.txt").write_text(f"{wall:.3f}\n")
    if force_record_bindings:
        b = SCALING / "bindings" / f"{group}_{e['key']}.txt"
        if not b.exists():
            b.parent.mkdir(parents=True, exist_ok=True)
            b.write_text(" ".join(cmd) + "\n" + r.stderr)
    return wall, True


def run_group(group, entries):
    nreps = group_reps(group)
    # warm-up pass (one per entry), then repetitions with rotated order
    for e in entries:
        w, ran = run_one(group, e, "warmup")
        if ran:
            print(f"[{group}] warmup {e['key']}: {w:.1f} s", flush=True)
    for r in range(nreps):
        for e in rotated(entries, r):
            w, ran = run_one(group, e, f"rep{r}")
            if ran:
                print(f"[{group}] rep{r} {e['key']}: {w:.1f} s", flush=True)


def cmd_probe(args):
    g = groups()
    probes = {}
    for name, key in (("strong_L6", "np1"), ("weak_200k", "np8")):
        e = next(x for x in g[name] if x["key"] == key)
        w, ran = run_one(name, e, "probe", force_record_bindings=False)
        wall = w if ran else float((run_dir(name, key, "probe") / "wall.txt").read_text())
        t = json.loads((run_dir(name, key, "probe") / "timings.json").read_text())
        probes[f"{name}/{key}"] = dict(wall=wall, solve=t["timings"]["solve"]["max"],
                                       iterations=t["solver"]["iterations"], dofs=t["dofs"])
        print(name, key, probes[f"{name}/{key}"], flush=True)
    (SCALING / "probe.json").write_text(json.dumps(probes, indent=1))


def cmd_estimate(args):
    """Warm up L4 and L5 at every np (the shape of wall time vs np), scale by the L6 probe."""
    g = groups()
    probes = json.loads((SCALING / "probe.json").read_text())
    wall = {}
    for name in ("strong_L4", "strong_L5"):
        for e in g[name]:
            run_one(name, e, "warmup")
            wall[(name, e["key"])] = float((run_dir(name, e["key"], "warmup") / "wall.txt").read_text())
    l6_1 = probes["strong_L6/np1"]["wall"]
    shape5 = {k: wall[("strong_L5", f"np{p}")] / wall[("strong_L5", "np1")] for p, k in
              [(p, f"np{p}") for p in NP_LIST]}
    per_pass = {}
    per_pass["strong_L4"] = sum(wall[("strong_L4", f"np{p}")] for p in NP_LIST)
    per_pass["strong_L5"] = sum(wall[("strong_L5", f"np{p}")] for p in NP_LIST)
    per_pass["strong_L6"] = l6_1 * sum(shape5.values())
    w10 = probes["weak_200k/np10"]["wall"]
    # weak: wall time grows mildly with p (iterations grow); bound each p by the p=10 probe
    per_pass["weak_200k"] = w10 * len(NP_LIST)
    per_pass["weak_50k"] = w10 * len(NP_LIST) / 2   # coarse bound: fixed I/O and CG per-iteration share
    # ParMETIS: 6 configurations of L5 at np 4, 8, 10
    per_pass["pm_L5"] = 2 * sum(wall[("strong_L5", f"np{p}")] for p in PM_NP)
    total = sum(per_pass.values()) * (REPS + 1)   # warm-up counts as one repetition
    est = dict(per_pass_seconds=per_pass, total_seconds=total, total_hours=total / 3600)
    (SCALING / "estimate.json").write_text(json.dumps(est, indent=1))
    print(json.dumps(est, indent=1))


def hardware_summary():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    lscpu = {}
    for line in sh("lscpu").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            lscpu[k.strip()] = v.strip()
    mem = re.search(r"MemTotal:\s+(\d+) kB", Path("/proc/meminfo").read_text())
    keep = ["Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)",
            "NUMA node(s)", "L1d cache", "L2 cache", "L3 cache", "Hypervisor vendor"]
    return dict(
        cpu={k: lscpu.get(k) for k in keep},
        memory_gib=round(int(mem.group(1)) / 2**20, 2) if mem else None,
        kernel=platform.release(), mpi=sh("mpiexec --version | head -1"),
        compiler=sh("g++ --version | head -1"),
        git_commit=sh(f"git -C {ROOT} rev-parse --short HEAD"),
        notes=("WSL2 under Hyper-V: synthetic topology of 10 cores x 2 threads; the hybrid "
               "P/E-core layout of the i7-12700H is hidden from the guest (6a findings); "
               "STREAM-like triad peaks near 19.6 GB/s at 4 ranks (6a)."))


def cmd_run(args):
    g = groups()
    names = args.groups or [n for n in g if n != "pm_L5"]
    triad_path = SCALING / "triad.json"
    triad = json.loads(triad_path.read_text()) if triad_path.exists() else {}
    for n in NP_LIST:
        if str(n) in triad:
            continue
        for attempt in range(3):   # best of three tries to avoid a transient dip
            r = subprocess.run(MPI_PREFIX[:-1] + ["-n", str(n), str(TRIAD)], capture_output=True,
                               text=True, cwd=ROOT)
            m = re.search(r"aggregate ([\d.]+) GB/s best, ([\d.]+) GB/s average", r.stdout)
            if not m:
                sys.exit(f"triad failed: {r.stdout}{r.stderr}")
            best, avg = float(m.group(1)), float(m.group(2))
            cur = triad.get(str(n))
            if cur is None or best > cur["best_gbs"]:
                triad[str(n)] = dict(best_gbs=best, avg_gbs=avg)
        print(f"triad np={n}: {triad[str(n)]}", flush=True)
        triad_path.write_text(json.dumps(triad, indent=1))
    for name in names:
        run_group(name, g[name])


# ----------------------------------------------------------------------------- aggregate
PHASES = ["read", "partition", "distribute", "assembly", "bcs", "solve", "postprocess", "output",
          "total"]
COMPONENTS = ["spmv_local", "halo_post", "halo_wait", "spmv_ghost", "dot_local", "allreduce",
              "vector_ops", "setup", "unaccounted", "total"]


def extract(t):
    """Raw per-run values from one timings.json."""
    ld = t["load"]
    rows, nnz = ld["owned_rows"], ld["local_nnz"]
    bytes_spmv = sum(z * 12 + (r + 1) * 4 + r * 16 for r, z in zip(rows, nnz))
    bd = t["solve_breakdown"]
    it = t["solver"]["iterations"]
    spmv_time = bd["spmv_local"]["max"] + bd["spmv_ghost"]["max"]
    run = dict(
        iterations=it, dofs=t["dofs"], nnz=t["nnz"], edge_cut=t["partition"]["edge_cut"],
        phases={k: t["timings"][k]["max"] for k in PHASES},
        phases_mean={k: t["timings"][k]["mean"] for k in PHASES},
        breakdown_max={k: bd[k]["max"] for k in COMPONENTS},
        breakdown_mean={k: bd[k]["mean"] for k in COMPONENTS},
        time_per_iteration=bd["total"]["max"] / it,
        owned_rows_max=max(rows), owned_rows_mean=sum(rows) / len(rows),
        nnz_max_over_mean=ld["local_nnz_max_over_mean"],
        halo=ld["halo"], neighbors=ld["neighbors"],
        halo_bytes_per_iteration_max=bd["halo_bytes_sent_per_iteration"]["max"],
        spmv_bytes=bytes_spmv, spmv_time_max=spmv_time,
        spmv_bandwidth_gbs=bytes_spmv * it / spmv_time / 1e9 if spmv_time > 0 else None,
        solve=t["timings"]["solve"]["max"], total=t["timings"]["total"]["max"])
    return run


def stats(values):
    v = [x for x in values if x is not None]
    med = statistics.median(v)
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    mean = statistics.fmean(v)
    return dict(median=med, min=min(v), max=max(v), mean=mean, cv=sd / mean if mean else 0.0)


def summarize(runs):
    keys = ["solve", "total", "time_per_iteration", "spmv_bandwidth_gbs", "iterations",
            "spmv_time_max"]
    s = {k: stats([r[k] for r in runs]) for k in keys}
    s["phases"] = {k: stats([r["phases"][k] for r in runs]) for k in PHASES}
    s["breakdown_mean"] = {k: stats([r["breakdown_mean"][k] for r in runs]) for k in COMPONENTS}
    s["breakdown_max"] = {k: stats([r["breakdown_max"][k] for r in runs]) for k in COMPONENTS}
    s["noisy"] = s["solve"]["cv"] > 0.05
    return s


def load_group(name, entries):
    out = {}
    for e in entries:
        runs = []
        for r in range(group_reps(name)):
            f = run_dir(name, e["key"], f"rep{r}") / "timings.json"
            if f.exists():
                runs.append(extract(json.loads(f.read_text())))
        if runs:
            out[e["key"]] = dict(np=e["np"], partitioner=e["partitioner"], runs=runs,
                                 summary=summarize(runs))
    return out


def karp_flatt(S, p):
    return (1 / S - 1 / p) / (1 - 1 / p) if p > 1 else None


def cmd_aggregate(args):
    g = groups()
    res = dict(hardware=hardware_summary(), settings=dict(
        path="distributed", partitioner="metis", instrument=True, plane="plane_stress",
        rtol=1e-10, mpiexec_flags=" ".join(MPI_PREFIX[1:]), repetitions={n: group_reps(n) for n in g}, warmups=1,
        np_list={n: group_np(n) for n in g},
        spmv_traffic_model=("bytes per SpMV = sum over ranks of nnz*(8+4) + (rows+1)*4 + rows*8 "
                            "(x) + rows*8 (y); approximate lower bound, x assumed read once; "
                            "time = max-over-ranks spmv_local + max-over-ranks spmv_ghost"),
        derived_basis="median over repetitions; solve time is the max over ranks"))
    tf = SCALING / "triad.json"
    res["triad"] = json.loads(tf.read_text()) if tf.exists() else {}
    res["bindings"] = {}
    for f in sorted((SCALING / "bindings").glob("*.txt")) if (SCALING / "bindings").exists() else []:
        res["bindings"][f.stem] = f.read_text()
    mp = SCALING / "meshes.json"
    res["meshes"] = json.loads(mp.read_text()) if mp.exists() else {}
    res["groups"] = {}
    for name, entries in g.items():
        data = load_group(name, entries)
        if data:
            res["groups"][name] = data
    # derived metrics
    for name, data in res["groups"].items():
        if name.startswith("strong_") and "np1" in data:
            base = data["np1"]["summary"]
            for key, d in data.items():
                p, s = d["np"], d["summary"]
                for what in ("solve", "total"):
                    S = base[what]["median"] / s[what]["median"]
                    d.setdefault("derived", {})[what] = dict(
                        speedup=S, efficiency=S / p, karp_flatt=karp_flatt(S, p),
                        speedup_lo=base[what]["median"] / s[what]["max"],
                        speedup_hi=base[what]["median"] / s[what]["min"])
        if name.startswith("weak_") and "np1" in data:
            t1 = data["np1"]["summary"]["time_per_iteration"]["median"]
            for key, d in data.items():
                ti = d["summary"]["time_per_iteration"]["median"]
                d["derived"] = dict(weak_efficiency_per_iteration=t1 / ti,
                                    dofs_per_rank=d["runs"][0]["dofs"] / d["np"])
        if name == "pm_L5":
            for key, d in data.items():
                d["derived"] = dict(
                    edge_cut=d["runs"][0]["edge_cut"],
                    owned_rows_max_over_mean=statistics.median(
                        r["owned_rows_max"] / r["owned_rows_mean"] for r in d["runs"]),
                    partition_time_median=d["summary"]["phases"]["partition"]["median"])
    out = ROOT / "docs" / "benchmark_results.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out} ({len(res['groups'])} groups)")
    noisy = [(n, k, d["summary"]["solve"]["cv"]) for n, dd in res["groups"].items()
             for k, d in dd.items() if d["summary"]["noisy"]]
    print("noisy (CV > 5 %):", noisy)


def cmd_parmetis(args):
    if not EXE_PM.exists():
        sys.exit(f"{EXE_PM} missing: cmake -S . -B build-parmetis -DUSE_PARMETIS=ON && build it")
    run_group("pm_L5", groups()["pm_L5"])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for n in ("meshes", "probe", "estimate", "aggregate", "parmetis"):
        sub.add_parser(n)
    r = sub.add_parser("run")
    r.add_argument("groups", nargs="*", help="subset of groups (default: strong and weak)")
    args = p.parse_args()
    SCALING.mkdir(parents=True, exist_ok=True)
    globals()["cmd_" + args.cmd](args)


if __name__ == "__main__":
    main()
