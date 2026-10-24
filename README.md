# mpi-cst-fem-solver

A parallel 2D linear-elastic finite element solver in C++ using MPI domain decomposition and Constant Strain Triangle (CST) elements. A Python toolchain (Gmsh API) generates meshes and drives benchmarks of parallel speedup and scalability.

Status: Milestone 7b (optional PETSc backend and its GAMG benchmark; verified against the Kirsch solution). See [docs/SPEC.md](docs/SPEC.md) and the verification report [docs/VERIFICATION.md](docs/VERIFICATION.md) (convergence rates, figures).

## Build

Requirements: CMake >= 3.20, a C++17 compiler, MPI, METIS (`libmetis-dev`), Python >= 3.10 for the tooling.

```
cmake -S . -B build
cmake --build build -j
```

Optional backends (both OFF by default):

```
cmake -S . -B build -DUSE_PARMETIS=ON   # ParMETIS parallel partitioner
cmake -S . -B build -DUSE_PETSC=ON      # PETSc solver backend (via pkg-config)
```

PETSc is found with pkg-config (module `PETSc`); besides the default search path, CMake also looks in `PETSC_DIR[/PETSC_ARCH]/lib/pkgconfig` (CMake or environment variable) and in the Debian/Ubuntu prefix `/usr/lib/petscdir/petsc-real`. Configuration fails unless PETSc is a real double-precision build (`PetscScalar = double`) whose `PetscInt` holds our `int` indices (32-bit PETSc indices match; 64-bit ones are a safe widening). On Ubuntu 24.04: `sudo apt-get install petsc-dev pkgconf` (PETSc 3.19.6, real, 32-bit indices, built against the system OpenMPI). CI builds without PETSc.

A custom METIS location can be given with `-DMETIS_ROOT=/path`.

## Run

From the repository root, after generating the meshes (see "Mesh generation"):

```
mpiexec -n 1 ./build/cstfem_solve cases/kirsch_L2.case     # serial path
mpiexec -n 4 ./build/cstfem_solve cases/kirsch_L2.case     # distributed path, 4 ranks
```

Case keys that select how the problem is run:

| Key | Values | Meaning |
|---|---|---|
| `path` | `auto` (default), `serial`, `distributed` | `auto`: serial code for 1 rank, distributed otherwise. `distributed` on 1 rank runs the full distributed code path (the fair baseline for speedup studies). `serial` with more than 1 rank is an error. |
| `partitioner` | `metis` (default), `parmetis` | METIS on rank 0, or ParMETIS on all ranks (needs `-DUSE_PARMETIS=ON`). |
| `solver` | `cg_jacobi` (default), `petsc` | In-house Jacobi-preconditioned CG, or the PETSc backend (needs `-DUSE_PETSC=ON`): KSPCG with the unpreconditioned residual norm, relative tolerance `rtol`, absolute tolerance 1e-300, `max_iter`, and PCGAMG by default. `petsc` always runs the distributed code path (also on 1 rank; `path = serial` is an error). |
| `petsc.near_nullspace` | `true` (default), `false` | Attach the three rigid-body modes, built from the owned node coordinates (`MatNullSpaceCreateRigidBody`), as the near-null space for GAMG. |
| `petsc.options` | PETSc options, unquoted, e.g. `-pc_type jacobi` | Applied last through `KSPSetFromOptions`, so they override the defaults (e.g. `-pc_type jacobi` for the cross-check with the in-house solver, `-ksp_view`). They live in a temporary options database for that solve, together with `PETSC_OPTIONS` from the environment. |
| `petsc.log_view` | `false` (default), `true` | Write PETSc's `-log_view` performance summary to `<output_dir>/petsc_log_view.txt`. |
| `instrument` | `true` (default), `false` | Per-rank time breakdown of the solver (SpMV local/ghost, halo post/wait, dot products, Allreduce, vector operations, halo bytes and messages per iteration), written to `timings.json` under `solve_breakdown`. The solution is identical either way. |

Outputs in the case's `output_dir` (e.g. `results/kirsch_L2/`), for both paths:

- `solution.pvtu` with one `solution_<rank>.vtu` piece per rank (open the `.pvtu` in ParaView). Points: `displacement`, `owned` (1 on the owning rank; nodes shared between pieces appear in each piece that uses them, so filter `owned = 1` to remove duplicates), `gmsh_tag`. Cells: `sigma_xx`, `sigma_yy`, `tau_xy`, `von_mises`, `sigma_zz` (plane strain), `partition_id`, `element_id`, and for the Kirsch case the analytical stresses and the errors.
- `hole_sigma_tt.csv`, `section_x0_sigma_xx.csv` (Kirsch case; gathered on rank 0 in parallel).
- `timings.json`: path, partitioner, ranks, DOFs, nnz, solver iterations and residuals, error metrics; for the distributed path also per-phase wall times (min/max/mean over ranks: read, partition, distribute, assembly, bcs, solve, postprocess, output, total) and the per-rank load report. With `solver = petsc`, `timings.json` also has the phases `petsc_matrix` (MATMPIAIJ, vectors and near-null space from our CSR), `pc_setup` (KSPSetUp + PCSetUp) and `ksp_solve` (KSPSolve), and the solver section records `ksp_type`, `pc_type`, `near_nullspace` and `petsc_options`. The true residual `||b - Ax|| / ||b||` is recomputed with our own distributed operator for both backends; `solve_breakdown` is only written for `cg_jacobi`.

`python/check_vtu.py <output_dir> [--compare <reference_dir>]` decodes all pieces independently (xml + base64 + numpy), and optionally compares displacements (by Gmsh tag), element arrays (by element id) and CSVs with another run.

The partitioning step can also be run alone:

```
mpiexec -n 4 ./build/cstfem_solve cases/kirsch_L3.case --partition-only
```

which prints a partition report and writes `partition.pvtu` with one piece per rank (`partition_id` per cell, `owner_rank` per node).

## Benchmarks

`build/benchmarks/stream_triad [elements] [repetitions]` is a STREAM-like triad: each MPI rank runs it on its own arrays at the same time, e.g. `mpiexec -n 4 --bind-to core --map-by core build/benchmarks/stream_triad`, to see where memory bandwidth saturates.

The strong and weak scaling study (METIS vs ParMETIS, SpMV bandwidth, analysis) is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md); `python/run_scaling.py` reproduces it.

On the single laptop node used here, PETSc CG + GAMG (rigid-body near-null space) keeps the iteration count flat (22-26 from 50k to 2.0 M DOFs, against 2,168-9,122 for CG + Jacobi, which grows as DOFs^0.51) and is 1.7-17 times faster in total time and 3.4-43 times in solve-only time than the in-house CG + Jacobi. Its parallel speedup is poor on this node (solve-only speedup 2.2-3.0 at best, total-time speedup 1.5-1.8 at best), because the iterations are memory-bandwidth bound and the rank-0 serial phases (read + partition) grow to 39-42 % of the total time at p = 8-10. Multi-node behavior was not tested. Details, tables and figures: the "PETSc GAMG" section of [docs/BENCHMARKS.md](docs/BENCHMARKS.md); `python/run_scaling.py gamg-*` reproduces it (needs a `-DUSE_PETSC=ON` build in `build-gamg`).

## Tests

```
ctest --test-dir build --output-on-failure
```

On OpenMPI with fewer physical cores than ranks, configure with `-DCSTFEM_MPI_OVERSUBSCRIBE=ON`.

Builds with `-DUSE_PETSC=ON` also register `test_petsc_np{1,2,4}` (PETSc CG + Jacobi vs the in-house CG + Jacobi on `kirsch_L1`, PETSc CG + GAMG with and without the near-null space vs the in-house solution, patch tests with GAMG for both plane conditions, Kirsch error metrics) and an end-to-end `solver = petsc` run on 2 ranks compared with the serial output.

## Mesh generation

The Kirsch quarter-plate meshes are generated with the Gmsh Python API (Python >= 3.10).

```
python3 -m venv python/.venv
python/.venv/bin/pip install -r python/requirements.txt

# level n halves the element size n times; output is Gmsh .msh 4.1 ASCII
python/.venv/bin/python python/mesh_kirsch.py --level 2 -o meshes/kirsch_L2.msh

# independent check (does not use gmsh): header, groups, orientation, geometry, quality
python/.venv/bin/python python/check_mesh.py meshes/kirsch_L2.msh
```

The convergence study (levels 0-5, figures in `docs/figures/`) is run with `python/run_convergence.py --np 8`; the analytical displacement field is derived and checked with `python/derive_kirsch_displacement.py` (needs `python/requirements-dev.txt`, i.e. sympy).

`mesh_kirsch.py --help` lists the parameters (`--a`, `--L`, `--h-hole`, `--h-far`, `--level`, `--scale`). `--scale s` multiplies both mesh sizes by `s` (combinable with `--level`); `--target-dofs N` searches `s` so that the mesh has `N` DOFs within ±3 % (for weak-scaling meshes). A small level-0 mesh is committed as `tests/data/kirsch_L0.msh` for the reader tests.

## Third-party dependencies

No third-party source code is vendored; dependencies are found on the system. The licenses below were read from the Debian/Ubuntu copyright files of the packages installed on the development machine (Ubuntu 24.04: `/usr/share/doc/<package>/copyright`); check them again for other versions.

| Dependency | Use | License (from the package copyright file) |
|---|---|---|
| OpenMPI 4.1.6 (`libopenmpi-dev`) | required (any MPI) | BSD-style 3-clause license text (redistribution with notice; no endorsement). Bundled components carry their own terms, e.g. ROMIO (University of Chicago permissive notice), ptmalloc2 (LGPL-2.1+), libltdl (LGPL-2+ with exception). |
| METIS 5.1.0 (`libmetis-dev`) | required | Apache-2.0. Some bundled GKlib files: BSD-3-clause (`ms_inttypes.h`, `ms_stdint.h`), LGPL-2.1+ (`getopt.c`, `gk_mksort.h`), BSD-like (`random.c`). |
| ParMETIS 4.0.3 (`libparmetis-dev`, Ubuntu multiverse) | optional (`-DUSE_PARMETIS=ON`) | The copyright file describes it as "non-free software" and states: "It can be freely used for educational and research purposes by non-profit institutions and US government agencies only. Other organizations are allowed to use ParMETIS/METIS only for evaluation purposes, and any further uses will require prior approval. The software may not be sold or redistributed without prior approval." |
| PETSc 3.19.6 (`petsc-dev` → `libpetsc-real3.19-dev`) | optional (`-DUSE_PETSC=ON`) | BSD-2-clause: "Copyright: (c) 1991-2014, UChicago Argonne, LLC and the PETSc Development Team"; redistribution in source and binary form with or without modification is permitted if the copyright notice, the conditions and the disclaimer are retained (and reproduced in documentation for binary forms); provided "as is" without warranty. The file notes that the license does not apply to packages obtained with PETSc's `--download-package` option, and lists a few files under other terms (STREAM benchmark license, Expat, MIT, BSD-3-clause). The Debian build of PETSc links further libraries with their own licenses (hypre, MUMPS, SuperLU, SuperLU_dist, PT-Scotch, ScaLAPACK, SuiteSparse, Trilinos ML, HDF5, FFTW, ...); these were not reviewed here. |

The default build (MPI + METIS) links the system libraries dynamically. METIS is Apache-2.0 with some bundled GKlib files under BSD-3-clause and LGPL-2.1+; OpenMPI's bundled components also include LGPL-licensed parts. This repository does not vendor or redistribute any of these libraries, so the repository code itself remains MIT.

**ParMETIS notice.** ParMETIS is strictly optional and off by default. This project never redistributes it: no vendored source, no release binaries, no container images that include it, and it is not used in CI. By installing ParMETIS and building with `-DUSE_PARMETIS=ON` you accept its license terms yourself; read `/usr/share/doc/libparmetis-dev/copyright` (or the license of the version you install) before doing so.

## License

MIT, see [LICENSE](LICENSE).
