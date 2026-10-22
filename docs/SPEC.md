# mpi-cst-fem-solver — Technical Specification

Version 0.1 (draft for review) · Owner: FreddyJuka · License: MIT

## 1. Purpose

A parallel 2D linear-elastic finite element solver in C++ using MPI domain decomposition and Constant Strain Triangle (CST) elements. A Python toolchain (Gmsh API) generates meshes and drives benchmarks of parallel speedup and scalability.

### Goals
- A distributed-memory implementation written in-house: partitioning, DOF ownership, assembly, halo exchange and the Krylov solver.
- Verification against the Kirsch solution (plate with a circular hole), including mesh-convergence rates.
- Reproducible strong- and weak-scaling benchmarks with plots.

### Non-goals (v1)
- Higher-order elements, 3D, nonlinearity, dynamics.
- Parallel mesh I/O (rank 0 reads the mesh in v1).

## 2. Physics and element formulation

- 2D small-strain linear elasticity, isotropic material: `E`, `nu`, thickness `t`.
- Plane condition selectable from input: `plane_stress` | `plane_strain`.
- CST element: linear shape functions, constant `B` (3×6), constant stress per element.
- Element stiffness: `Ke = t · A · Bᵀ D B` (6×6).
- Constitutive matrices:
  - Plane stress: `D = E/(1-ν²) · [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]`
  - Plane strain: `D = E/((1+ν)(1-2ν)) · [[1-ν, ν, 0], [ν, 1-ν, 0], [0, 0, (1-2ν)/2]]`
- Out-of-plane stress `σzz = ν(σxx + σyy)` is reported in plane strain.
- Units: any consistent system; SI is used in examples.

## 3. Input

### 3.1 Mesh
- Gmsh `.msh` format v4.1, ASCII (binary may be added later). The reader is written in-house.
- Required content: `$Nodes`, `$Elements` (type 2 = 3-node triangle, type 1 = 2-node line), `$PhysicalNames`, `$Entities`.
- Boundary conditions reference physical groups by name.

### 3.2 Case file
A plain `key = value` text file (zero dependencies), for example:

```
mesh            = meshes/kirsch_L3.msh
plane           = plane_stress
E               = 210e9
nu              = 0.3
thickness       = 0.01
partitioner     = metis          # metis | parmetis
solver          = cg_jacobi      # cg_jacobi | petsc
rtol            = 1e-10
max_iter        = 20000
output_dir      = results/kirsch_L3
# boundary conditions
bc.dirichlet    = left_sym:ux=0 ; bottom_sym:uy=0
bc.traction     = kirsch_exact   # or group:tx=..,ty=..
kirsch.sigma_inf = 100e6
kirsch.a         = 0.1
```

## 4. Domain decomposition

- **Element-based (non-overlapping) partition** of the element dual graph.
- `partitioner = metis`: rank 0 reads the mesh and calls `METIS_PartMeshDual`, then elements and nodes are scattered.
- `partitioner = parmetis` (only available when built with `-DUSE_PARMETIS=ON`): rank 0 reads the mesh and block-distributes elements; `ParMETIS_V3_PartMeshKway` partitions in parallel; elements are then migrated with `MPI_Alltoallv`. If ParMETIS was not compiled in, selecting it is a clear runtime error.
- **Node ownership:** a node shared by several partitions is owned by the lowest rank among them. Each rank holds its owned nodes plus ghost (non-owned) nodes referenced by its elements.
- **Global numbering:** owned DOFs are renumbered contiguously per rank (rank r owns global range `[offset_r, offset_r + n_owned_r)`), 2 DOFs per node.
- A `partition_id` field per element is kept for output.

## 5. Assembly

- Distributed CSR, **row-owner layout**: each rank stores the rows of its owned DOFs.
- Each rank computes `Ke` for its local elements. Contributions to rows owned by other ranks are packed and sent to the owner once, with `MPI_Alltoallv`, then summed.
- Columns are split into local (owned) and ghost blocks: `A_r = [A_loc | A_ghost]`, which allows SpMV to overlap communication and computation.
- The communication pattern for ghost values is built once (neighbor lists, send/recv index maps) and reused, via persistent `MPI_Isend`/`MPI_Irecv` or `MPI_Neighbor_alltoallv` on a distributed graph communicator.

## 6. Boundary conditions

- **Dirichlet:** symmetric elimination (zero row and column, unit diagonal, RHS corrected with `-K_col · ū`), which preserves symmetric positive definiteness for CG. Column elimination of ghost columns is done consistently across ranks.
- **Neumann:** consistent nodal forces from traction on 2-node line elements of a physical group, either constant or evaluated from an analytical field (Kirsch exact tractions).

## 7. Linear solver

- Abstract interface `LinearSolver` (setup / solve / stats).
- **Default backend `cg_jacobi`:** preconditioned conjugate gradient written in-house.
  - Distributed SpMV with halo exchange; overlap of local SpMV and ghost-value communication.
  - Dot products and norms via `MPI_Allreduce`.
  - Jacobi (diagonal) preconditioner.
  - Stopping criterion: `‖r_k‖₂ / ‖b‖₂ < rtol`.
- **Optional backend `petsc`** (CMake `-DUSE_PETSC=ON`): KSPCG + PCGAMG, for comparison in the benchmarks.
- Expected behavior: CG+Jacobi iterations grow as O(h⁻¹), whereas GAMG keeps iterations roughly constant. Benchmarks therefore report time per iteration in addition to total time.

## 8. Verification: Kirsch problem

- Geometry: quarter of a square plate with a central hole of radius `a`, side `L` (default `L/a = 10`), using symmetry.
- Boundary conditions:
  - `left_sym` (x = 0): `ux = 0`; `bottom_sym` (y = 0): `uy = 0`.
  - `hole`: traction-free.
  - `right` and `top`: **exact Kirsch tractions** `t = σ·n` evaluated from the analytical field. This removes the finite-width error, so the discrete solution converges to the Kirsch solution.
- Analytical stresses (uniaxial `σ∞` along x, θ measured from x):
  - `σrr = σ∞/2 (1 − a²/r²) + σ∞/2 (1 − 4a²/r² + 3a⁴/r⁴) cos 2θ`
  - `σθθ = σ∞/2 (1 + a²/r²) − σ∞/2 (1 + 3a⁴/r⁴) cos 2θ`
  - `σrθ = −σ∞/2 (1 + 2a²/r² − 3a⁴/r⁴) sin 2θ`
- Metrics:
  - Relative L2 error of stress (element-wise, evaluated at the centroid).
  - Stress concentration factor at (r = a, θ = 90°), compared with the exact value of 3.
  - Mesh-convergence study over ≥ 4 refinement levels. Expected rate for CST stress error in L2: O(h).
- Displacement-based error norms are optional and will only be added once the analytical displacement field has been verified against a reference text.

## 9. Output

- Per-rank `.vtu` (XML, binary appended/base64) plus a `.pvtu` master file, written in-house with no external dependencies.
  - Point data: displacement `(ux, uy)`.
  - Cell data: `σxx, σyy, τxy, von Mises` (`σzz` in plane strain), `partition_id`.
- CSV files for verification: `σθθ` along the hole boundary, `σxx` along the x = 0 section, and analytical values next to them.
- `timings.json` per run: wall time per phase (read, partition, distribute, assembly, BCs, solve, output), iteration count, final residual, ranks, DOFs.

## 10. Python tooling (`python/`)

- `mesh_kirsch.py`: Gmsh API generator for the quarter plate. Parameters: `a`, `L`, mesh size near the hole, size growth, refinement level. Physical groups: `hole`, `left_sym`, `bottom_sym`, `right`, `top`, `domain`. Exports `.msh` v4.1 ASCII.
- `run_convergence.py`: runs the refinement levels and plots error vs h on log-log axes, with the fitted rate.
- `run_scaling.py`: strong scaling (fixed mesh, P = 1, 2, 4, …) and weak scaling (constant DOFs per rank). Plots speedup, efficiency, time per iteration and iteration count.
- Plots with matplotlib; the hardware (CPU model, cores, WSL2/native) is recorded in every benchmark result.

## 11. Repository layout

```
mpi-cst-fem-solver/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── docs/SPEC.md
├── include/cstfem/     # headers
├── src/                # mesh reader, partition, assembly, solver, io, main
├── tests/              # unit + MPI tests (CTest)
├── python/             # mesh generation, benchmarks, plotting
├── cases/              # example case files
└── .github/workflows/ci.yml
```

## 12. Build and dependencies

- Required: CMake ≥ 3.20, C++17, MPI (OpenMPI or MPICH), METIS.
- Optional (all OFF by default):
  - `-DUSE_PARMETIS=ON`: ParMETIS parallel partitioner.
  - `-DUSE_PETSC=ON`: PETSc solver backend (pkg-config).
- Python ≥ 3.10: `gmsh`, `numpy`, `matplotlib`.

### Licensing policy
- All code in this repository is MIT-licensed.
- No third-party source code is vendored; dependencies are found on the system via CMake.
- The default build (MPI + METIS) contains only permissively licensed dependencies. The METIS license must be confirmed for the installed version.
- ParMETIS has its own restrictive terms. It is strictly optional, is never redistributed (no vendored source, no release binaries, no container images that include it), and is not used in CI.
- The README includes a "Third-party dependencies" section listing each dependency, its license, and a notice that ParMETIS terms are accepted by the user when installing it. The license text of the installed ParMETIS version must be checked before writing that notice.

## 13. Testing

- Unit tests: CST `Ke` against a hand-computed element; symmetry and rigid-body null space of `Ke` (3 zero eigenvalues).
- **Patch test:** a linear displacement field must be reproduced exactly (to machine precision), in serial and in parallel.
- **Parallel consistency:** the solution with P = 1, 2, 4 ranks agrees to within the solver tolerance.
- Mesh reader test on a small committed `.msh` file.
- CTest runs MPI tests through `mpiexec -n N` (with `--oversubscribe` in CI).

## 14. CI

GitHub Actions on `ubuntu-24.04`: install `libopenmpi-dev` and `libmetis-dev`, configure with default options (ParMETIS and PETSc OFF), build, run `ctest`.

## 15. Milestones

1. Repository scaffold + CMake + CI skeleton
2. Python Kirsch mesh generator
3. Serial solver (reader, CST, assembly, BCs, CG, VTU) + unit and patch tests
4. MPI: partitioning, ownership, distributed assembly, distributed PCG, PVTU
5. Kirsch verification + convergence study
6. Strong/weak scaling benchmarks
7. Optional PETSc backend + comparison
8. README with figures and third-party license section, release v1.0 (source only)
