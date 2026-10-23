#pragma once

// Optional PETSc solver backend (`solver = petsc`, CMake -DUSE_PETSC=ON). The declarations
// carry no PETSc types, so callers compile without PETSc headers; the definitions exist only
// in builds with USE_PETSC (src/petsc_backend.cpp). Guard calls with `#if USE_PETSC`.

#include <mpi.h>

#include <string>
#include <vector>

#include "cstfem/distributed.hpp"
#include "cstfem/solver.hpp"

namespace cstfem {

// PetscInitialize on MPI_COMM_WORLD of an already initialized MPI (PETSc then leaves
// MPI_Finalize to the caller); options come only from the PETSC_OPTIONS environment variable
// and from solve_petsc. Call after MPI_Init; petsc_finalize() before MPI_Finalize.
void petsc_initialize();
void petsc_finalize();

struct PetscBackendOptions {
  bool near_nullspace = true;  // rigid-body modes from the owned node coordinates (for GAMG)
  std::string options;         // extra PETSc options, e.g. "-pc_type jacobi"
};

// Per-rank wall times and the solver objects actually used.
struct PetscSolveInfo {
  double matrix_time = 0.0;    // MATMPIAIJ, vectors and near-null space from our CSR
  double pc_setup_time = 0.0;  // KSPSetUp + PCSetUp (GAMG hierarchy, Jacobi diagonal)
  double ksp_solve_time = 0.0; // KSPSolve
  std::string ksp_type, pc_type;
  bool near_nullspace = false;
};

// Solves A x = b with KSPCG (unpreconditioned residual norm, relative tolerance rtol, absolute
// tolerance 1e-300, max_iter from `options`) and PCGAMG by default; `petsc.options` are applied
// last through KSPSetFromOptions, inside a temporary options database so that one solve does
// not change the next. A is our distributed matrix after Dirichlet elimination: owned rows
// are contiguous per rank, as PETSc expects; halo columns are mapped back to global indices.
// `owned_xy` holds the interleaved (x, y) of the owned nodes (owned DOF 2 n + c). x is
// resized to the owned rows. The returned stats follow pcg_jacobi: rel_residual is PETSc's
// final unpreconditioned residual norm / ||b||, true_rel_residual is ||b - A x|| / ||b||
// recomputed with our DistOperator. Collective.
SolverStats solve_petsc(const DistMatrix& A, const std::vector<double>& b, std::vector<double>& x,
                        const std::vector<double>& owned_xy, const SolverOptions& options,
                        const PetscBackendOptions& petsc, MPI_Comm comm,
                        PetscSolveInfo* info = nullptr);

// -log_view support: start PETSc's default logging, and write the log as text. Collective.
void petsc_log_begin();
void petsc_log_view(const std::string& path, MPI_Comm comm);

}  // namespace cstfem
