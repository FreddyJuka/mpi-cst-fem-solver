# Changelog

## v1.0.0

First release. Summary of the milestones:

1. **Milestone 1:** repository scaffold, CMake build and CI skeleton.
2. **Milestone 2:** Gmsh Kirsch mesh generator (`mesh_kirsch.py`) and an independent mesh checker.
3. **Milestone 3:** Gmsh reader, case-file parser, CST element, CSR assembly, Dirichlet and traction conditions, Jacobi-PCG, Kirsch verification field, VTU/CSV output and the serial executable.
4. **Milestone 4:** METIS partitioning, ownership and global numbering, distributed assembly and boundary conditions, halo exchange, parallel Jacobi-PCG, parallel `.pvtu`/CSV/`timings.json` output, optional ParMETIS.
5. **Milestone 5:** Kirsch verification and convergence study (levels 0-5, plane stress and plane strain), with the analytical displacement field derived symbolically.
6. **Milestone 6:** solver instrumentation, mesh scaling, hardware discovery, and the strong and weak scaling study (SpMV bandwidth vs STREAM triad, METIS vs ParMETIS, Karp-Flatt analysis).
7. **Milestone 7:** optional PETSc backend (CG + GAMG with the rigid-body near-null space), GAMG benchmark and comparison with the in-house solver, diagnosis of the bimodal L6 p = 1 time (not explained), monotonic timing in the scripts and `clock_check`.
8. **Release prep (8a):** final README, `kirsch_fields` figure, link checker, version 1.0.0.
