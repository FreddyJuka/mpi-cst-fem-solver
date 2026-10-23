#include "cstfem/petsc_backend.hpp"

#include <petscksp.h>

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace cstfem {

// The configure check in CMakeLists.txt tests the same; these keep the code honest if that
// check is bypassed (e.g. a different PETSc found at build time).
static_assert(std::is_same<PetscScalar, double>::value,
              "the PETSc backend needs a real, double-precision PETSc (PetscScalar = double)");
static_assert(std::is_same<PetscReal, double>::value, "PetscReal must be double");
static_assert(std::is_integral<PetscInt>::value && std::is_signed<PetscInt>::value &&
                  sizeof(PetscInt) >= sizeof(int),
              "PetscInt must hold every int index (32- or 64-bit PETSc indices)");

namespace {

void check(PetscErrorCode ierr, const char* call) {
  if (ierr != PETSC_SUCCESS)
    throw std::runtime_error(std::string("PETSc error ") + std::to_string(static_cast<int>(ierr)) +
                             " in " + call);
}
#define CSTFEM_PETSC(call) check((call), #call)

// Destroys PETSc objects on scope exit (also on exceptions).
struct Objects {
  Mat A = nullptr;
  Vec x = nullptr, b = nullptr, coords = nullptr;
  MatNullSpace nsp = nullptr;
  KSP ksp = nullptr;
  ~Objects() {
    KSPDestroy(&ksp);
    MatNullSpaceDestroy(&nsp);
    VecDestroy(&coords);
    VecDestroy(&b);
    VecDestroy(&x);
    MatDestroy(&A);
  }
};

// Makes `options` (after PETSC_OPTIONS from the environment) the default options database
// while it lives, so options of one solve do not leak into the next.
class ScopedOptions {
public:
  explicit ScopedOptions(const std::string& options) {
    CSTFEM_PETSC(PetscOptionsCreate(&db_));
    if (const char* env = std::getenv("PETSC_OPTIONS"))
      CSTFEM_PETSC(PetscOptionsInsertString(db_, env));
    CSTFEM_PETSC(PetscOptionsInsertString(db_, options.c_str()));
    CSTFEM_PETSC(PetscOptionsPush(db_));
  }
  ~ScopedOptions() {
    PetscOptionsPop();
    PetscOptionsDestroy(&db_);
  }
  ScopedOptions(const ScopedOptions&) = delete;
  ScopedOptions& operator=(const ScopedOptions&) = delete;

private:
  PetscOptions db_ = nullptr;
};

void copy_in(Vec v, const std::vector<double>& values) {
  PetscScalar* a = nullptr;
  CSTFEM_PETSC(VecGetArray(v, &a));
  for (std::size_t i = 0; i < values.size(); ++i) a[i] = values[i];
  CSTFEM_PETSC(VecRestoreArray(v, &a));
}

}  // namespace

void petsc_initialize() {
  PETSC_COMM_WORLD = MPI_COMM_WORLD;
  CSTFEM_PETSC(PetscInitializeNoArguments());
}

void petsc_finalize() { CSTFEM_PETSC(PetscFinalize()); }

void petsc_log_begin() { CSTFEM_PETSC(PetscLogDefaultBegin()); }

void petsc_log_view(const std::string& path, MPI_Comm comm) {
  PetscViewer viewer = nullptr;
  CSTFEM_PETSC(PetscViewerASCIIOpen(comm, path.c_str(), &viewer));
  const PetscErrorCode ierr = PetscLogView(viewer);
  PetscViewerDestroy(&viewer);
  check(ierr, "PetscLogView");
}

SolverStats solve_petsc(const DistMatrix& A, const std::vector<double>& b, std::vector<double>& x,
                        const std::vector<double>& owned_xy, const SolverOptions& options,
                        const PetscBackendOptions& petsc, MPI_Comm comm, PetscSolveInfo* info) {
  const int n = A.n_owned;
  if (static_cast<int>(b.size()) != n || static_cast<int>(owned_xy.size()) != n)
    throw std::runtime_error("solve_petsc: b and owned_xy must have one entry per owned row");
  if (n % 2 != 0) throw std::runtime_error("solve_petsc: odd number of owned rows (block size 2)");
  PetscSolveInfo local_info;
  PetscSolveInfo& inf = info ? *info : local_info;
  ScopedOptions scoped(petsc.options);
  Objects o;

  // ---- matrix: MATMPIAIJ, block size 2, our row ownership
  double t0 = MPI_Wtime();
  CSTFEM_PETSC(MatCreate(comm, &o.A));
  CSTFEM_PETSC(MatSetSizes(o.A, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  CSTFEM_PETSC(MatSetType(o.A, MATMPIAIJ));
  CSTFEM_PETSC(MatSetBlockSize(o.A, 2));
  std::vector<PetscInt> d_nnz(n), o_nnz(n);
  for (int i = 0; i < n; ++i) {
    d_nnz[i] = A.loc.row_ptr[i + 1] - A.loc.row_ptr[i];
    o_nnz[i] = A.ghost.row_ptr[i + 1] - A.ghost.row_ptr[i];
  }
  CSTFEM_PETSC(MatMPIAIJSetPreallocation(o.A, 0, d_nnz.data(), 0, o_nnz.data()));
  PetscInt row_begin = 0, row_end = 0;
  CSTFEM_PETSC(MatGetOwnershipRange(o.A, &row_begin, &row_end));
  if (row_begin != A.first_row || row_end != A.first_row + n)
    throw std::runtime_error("solve_petsc: PETSc row ownership differs from ours");
  std::vector<PetscInt> cols;
  std::vector<PetscScalar> vals;
  for (int i = 0; i < n; ++i) {
    cols.clear();
    vals.clear();
    for (int k = A.loc.row_ptr[i]; k < A.loc.row_ptr[i + 1]; ++k) {
      cols.push_back(static_cast<PetscInt>(A.first_row) + A.loc.cols[k]);
      vals.push_back(A.loc.vals[k]);
    }
    for (int k = A.ghost.row_ptr[i]; k < A.ghost.row_ptr[i + 1]; ++k) {
      cols.push_back(A.halo_globals[A.ghost.cols[k]]);
      vals.push_back(A.ghost.vals[k]);
    }
    const PetscInt row = static_cast<PetscInt>(A.first_row) + i;
    CSTFEM_PETSC(MatSetValues(o.A, 1, &row, static_cast<PetscInt>(cols.size()), cols.data(),
                              vals.data(), INSERT_VALUES));
  }
  CSTFEM_PETSC(MatAssemblyBegin(o.A, MAT_FINAL_ASSEMBLY));
  CSTFEM_PETSC(MatAssemblyEnd(o.A, MAT_FINAL_ASSEMBLY));
  // Symmetric elimination keeps the stiffness matrix symmetric positive definite.
  CSTFEM_PETSC(MatSetOption(o.A, MAT_SYMMETRIC, PETSC_TRUE));
  CSTFEM_PETSC(MatSetOption(o.A, MAT_SPD, PETSC_TRUE));

  CSTFEM_PETSC(MatCreateVecs(o.A, &o.x, &o.b));  // block size 2 from the matrix
  copy_in(o.b, b);
  inf.near_nullspace = petsc.near_nullspace;
  if (petsc.near_nullspace) {
    CSTFEM_PETSC(MatCreateVecs(o.A, &o.coords, nullptr));
    copy_in(o.coords, owned_xy);
    CSTFEM_PETSC(MatNullSpaceCreateRigidBody(o.coords, &o.nsp));
    CSTFEM_PETSC(MatSetNearNullSpace(o.A, o.nsp));
  }
  inf.matrix_time = MPI_Wtime() - t0;

  // ---- KSP: CG on the unpreconditioned residual, GAMG unless overridden by options
  t0 = MPI_Wtime();
  CSTFEM_PETSC(KSPCreate(comm, &o.ksp));
  CSTFEM_PETSC(KSPSetOperators(o.ksp, o.A, o.A));
  CSTFEM_PETSC(KSPSetType(o.ksp, KSPCG));
  CSTFEM_PETSC(KSPSetNormType(o.ksp, KSP_NORM_UNPRECONDITIONED));
  CSTFEM_PETSC(KSPSetTolerances(o.ksp, options.rtol, 1e-300, PETSC_DEFAULT, options.max_iter));
  PC pc = nullptr;
  CSTFEM_PETSC(KSPGetPC(o.ksp, &pc));
  CSTFEM_PETSC(PCSetType(pc, PCGAMG));
  CSTFEM_PETSC(KSPSetFromOptions(o.ksp));
  CSTFEM_PETSC(KSPSetUp(o.ksp));
  CSTFEM_PETSC(KSPGetPC(o.ksp, &pc));
  CSTFEM_PETSC(PCSetUp(pc));
  inf.pc_setup_time = MPI_Wtime() - t0;
  KSPType ktype = nullptr;
  PCType ptype = nullptr;
  CSTFEM_PETSC(KSPGetType(o.ksp, &ktype));
  CSTFEM_PETSC(PCGetType(pc, &ptype));
  inf.ksp_type = ktype ? ktype : "";
  inf.pc_type = ptype ? ptype : "";

  t0 = MPI_Wtime();
  CSTFEM_PETSC(KSPSolve(o.ksp, o.b, o.x));
  inf.ksp_solve_time = MPI_Wtime() - t0;

  // ---- results back into our owned vector, stats as in pcg_jacobi
  x.assign(n, 0.0);
  const PetscScalar* xa = nullptr;
  CSTFEM_PETSC(VecGetArrayRead(o.x, &xa));
  for (int i = 0; i < n; ++i) x[i] = xa[i];
  CSTFEM_PETSC(VecRestoreArrayRead(o.x, &xa));

  SolverStats st;
  PetscInt its = 0;
  PetscReal rnorm = 0.0;
  KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
  CSTFEM_PETSC(KSPGetIterationNumber(o.ksp, &its));
  CSTFEM_PETSC(KSPGetResidualNorm(o.ksp, &rnorm));
  CSTFEM_PETSC(KSPGetConvergedReason(o.ksp, &reason));
  const DistReduction reduction(comm);
  const double bnorm = reduction.norm(b);
  st.iterations = static_cast<int>(its);
  st.converged = reason > 0;
  st.rel_residual = bnorm > 0.0 ? rnorm / bnorm : 0.0;
  std::ostringstream msg;
  msg << (st.converged ? "converged" : "NOT converged") << " (" << KSPConvergedReasons[reason]
      << ") in " << st.iterations << " iterations, relative residual " << st.rel_residual
      << " (rtol " << options.rtol << "), " << inf.ksp_type << " + " << inf.pc_type;
  st.message = msg.str();
  if (bnorm > 0.0) {
    std::vector<double> r;
    DistOperator(A).apply(x, r);
    for (int i = 0; i < n; ++i) r[i] = b[i] - r[i];
    st.true_rel_residual = reduction.norm(r) / bnorm;
  }
  if (st.converged && st.true_rel_residual > 10.0 * options.rtol) {
    std::ostringstream w;
    w << "WARNING: PETSc " << inf.ksp_type << " converged (" << st.rel_residual
      << ") but the true residual ||b - Ax|| / ||b|| = " << st.true_rel_residual
      << " exceeds 10 * rtol = " << 10.0 * options.rtol;
    st.warning = w.str();
  }
  return st;
}

}  // namespace cstfem
