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

```
mpiexec -n 2 ./build/cstfem_solve
```

## Tests

```
ctest --test-dir build --output-on-failure
```

On OpenMPI with fewer physical cores than ranks, configure with `-DCSTFEM_MPI_OVERSUBSCRIBE=ON`.

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
