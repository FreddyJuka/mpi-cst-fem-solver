# mpi-cst-fem-solver

A parallel 2D linear-elastic finite element solver in C++ using MPI domain decomposition and Constant Strain Triangle (CST) elements. A Python toolchain (Gmsh API) generates meshes and drives benchmarks of parallel speedup and scalability.

Status: Milestone 1 (scaffold). See [docs/SPEC.md](docs/SPEC.md).

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

A custom METIS location can be given with `-DMETIS_ROOT=/path`.

## Run

From the repository root, after generating the meshes (see "Mesh generation"):

```
mpiexec -n 1 ./build/cstfem_solve cases/kirsch_L2.case
```

Outputs go to the case's `output_dir` (e.g. `results/kirsch_L2/`): `solution.pvtu` + `solution_0.vtu` (open the `.pvtu` in ParaView), `timings.json`, and for the Kirsch case `hole_sigma_tt.csv` and `section_x0_sigma_xx.csv`. `python/check_vtu.py <output_dir>` decodes and checks the VTU output independently.

With more than one rank the solve is distributed (METIS partition, row-owner matrix, halo exchange, parallel Jacobi-PCG):

```
mpiexec -n 4 ./build/cstfem_solve cases/kirsch_L3.case
```

Rank 0 prints iterations, residuals, error metrics and a per-rank load report; parallel file output (VTU/CSV/timings) is not written yet (Milestone 4c). The partitioning step can also be run alone:

```
mpiexec -n 4 ./build/cstfem_solve cases/kirsch_L3.case --partition-only
```

which prints a partition report (METIS) and writes `partition.pvtu` with one piece per rank (`partition_id` per cell, `owner_rank` per node).

## Tests

```
ctest --test-dir build --output-on-failure
```

On OpenMPI with fewer physical cores than ranks, configure with `-DCSTFEM_MPI_OVERSUBSCRIBE=ON`.

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

`mesh_kirsch.py --help` lists the parameters (`--a`, `--L`, `--h-hole`, `--h-far`, `--level`). A small level-0 mesh is committed as `tests/data/kirsch_L0.msh` for the reader tests.

## Third-party dependencies

| Dependency | Use | License |
|---|---|---|
| MPI (e.g. OpenMPI, MPICH) | required | TBD — to be verified |
| METIS | required | TBD — to be verified |
| ParMETIS | optional (`-DUSE_PARMETIS=ON`) | TBD — to be verified |
| PETSc | optional (`-DUSE_PETSC=ON`) | TBD — to be verified |

No third-party source code is vendored; dependencies are found on the system.

## License

MIT, see [LICENSE](LICENSE).
