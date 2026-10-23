// MPI tests of the PETSc backend (only built with -DUSE_PETSC=ON) against the in-house
// distributed Jacobi-PCG. Run with any number of ranks; both solvers see the same partition.

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "cstfem/bc.hpp"
#include "cstfem/distributed.hpp"
#include "cstfem/kirsch.hpp"
#include "cstfem/petsc_backend.hpp"
#include "cstfem/postprocess.hpp"

using namespace cstfem;

namespace {

int g_rank = 0, g_size = 1, g_failures = 0;
const double kE = 210e9, kNu = 0.3, kT = 0.01;
const KirschParams kKirsch{100e6, 0.1};
const SolverOptions kOpt{1e-12, 20000, false};

#define CHECK(cond)                                                                            \
  do {                                                                                         \
    if (!(cond)) {                                                                             \
      std::fprintf(stderr, "[rank %d] %s:%d: CHECK failed: %s\n", g_rank, __FILE__, __LINE__, \
                   #cond);                                                                     \
      ++g_failures;                                                                            \
    }                                                                                          \
  } while (0)

std::string data(const char* name) { return std::string(CSTFEM_TEST_DATA_DIR) + "/" + name; }

double max_all(double v) {
  double m = 0.0;
  MPI_Allreduce(&v, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return m;
}

// max |a - b| / max |b| over all ranks.
double rel_max_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0, r = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d = std::max(d, std::abs(a[i] - b[i]));
    r = std::max(r, std::abs(b[i]));
  }
  return max_all(d) / max_all(r);
}

double rel(double a, double b) { return std::abs(a / b - 1.0); }

// Distributed problem after Dirichlet elimination, ready for either solver.
struct Problem {
  DistributedMesh dm;
  DofLayout layout;
  DistMatrix A;
  std::vector<double> b, owned_xy;
  PlaneCondition plane = PlaneCondition::Stress;
};

using Bcs = std::function<void(const Mesh&, DirichletDofs&, std::vector<double>&)>;

Problem build(const std::string& path, PlaneCondition plane, const Bcs& bcs) {
  Problem p;
  p.plane = plane;
  p.dm = read_partition_distribute(path, MPI_COMM_WORLD);
  p.layout = make_dof_layout(p.dm, MPI_COMM_WORLD);
  p.A = assemble_distributed(p.dm, p.layout, constitutive_matrix(kE, kNu, plane), kT,
                             MPI_COMM_WORLD);
  DirichletDofs local;
  std::vector<double> f_local(p.layout.n_local, 0.0);
  bcs(p.dm.local, local, f_local);
  p.b = accumulate_owned(p.layout, f_local);
  apply_dirichlet_distributed(p.dm, p.layout, local, p.A, p.b, MPI_COMM_WORLD);
  p.owned_xy.assign(p.dm.local.xy.begin(), p.dm.local.xy.begin() + 2 * p.dm.num_owned);
  return p;
}

Problem kirsch_problem(const char* mesh) {
  return build(data(mesh), PlaneCondition::Stress,
               [](const Mesh& m, DirichletDofs& d, std::vector<double>& f) {
                 add_dirichlet(m, parse_dirichlet("left_sym:ux=0 ; bottom_sym:uy=0"), d);
                 add_traction(m, "right", kT, kirsch_traction(kKirsch), f);
                 add_traction(m, "top", kT, kirsch_traction(kKirsch), f);
               });
}

SolverStats petsc_solve(const Problem& p, std::vector<double>& u, const PetscBackendOptions& o,
                        PetscSolveInfo* info = nullptr) {
  return solve_petsc(p.A, p.b, u, p.owned_xy, kOpt, o, MPI_COMM_WORLD, info);
}

KirschMetrics metrics(Problem& p, const std::vector<double>& u) {
  const std::vector<double> u_loc = to_mesh_local(p.layout, u);
  const Mat3 D = constitutive_matrix(kE, kNu, p.plane);
  const ElementStresses es = compute_element_stresses(p.dm.local, u_loc, D, kNu, p.plane);
  return kirsch_metrics_distributed(p.dm.local, u_loc, es, kKirsch, {kE, kNu, p.plane},
                                    MPI_COMM_WORLD);
}

// PETSc CG + Jacobi vs in-house CG + Jacobi: same algorithm, same stopping test.
void test_cross_check_jacobi() {
  Problem p = kirsch_problem("kirsch_L1.msh");
  std::vector<double> u_ref, u;
  const SolverStats sr = solve_distributed(p.A, p.b, u_ref, kOpt, MPI_COMM_WORLD);
  PetscSolveInfo info;
  const SolverStats sp = petsc_solve(p, u, {true, "-pc_type jacobi"}, &info);
  const double du = rel_max_diff(u, u_ref);
  if (g_rank == 0)
    std::printf("  kirsch_L1, rtol 1e-12: in-house CG+Jacobi %d iterations (true residual %.2e), "
                "PETSc %s+%s %d iterations (true residual %.2e); max rel. difference %.2e\n",
                sr.iterations, sr.true_rel_residual, info.ksp_type.c_str(), info.pc_type.c_str(),
                sp.iterations, sp.true_rel_residual, du);
  CHECK(sr.converged && sp.converged);
  CHECK(info.ksp_type == "cg" && info.pc_type == "jacobi");
  CHECK(du < 1e-8);
  CHECK(std::abs(sr.iterations - sp.iterations) <= 2);
}

// PETSc CG + GAMG (with and without the rigid-body near-null space) vs the in-house solution.
void test_gamg_vs_inhouse() {
  Problem p = kirsch_problem("kirsch_L1.msh");
  std::vector<double> u_ref;
  const SolverStats sr = solve_distributed(p.A, p.b, u_ref, kOpt, MPI_COMM_WORLD);
  CHECK(sr.converged);
  for (bool nsp : {true, false}) {
    std::vector<double> u;
    PetscSolveInfo info;
    const SolverStats sp = petsc_solve(p, u, {nsp, ""}, &info);
    const double du = rel_max_diff(u, u_ref);
    if (g_rank == 0)
      std::printf("  kirsch_L1: PETSc %s+%s, near-null space %s: %d iterations (in-house %d), "
                  "true residual %.2e, max rel. difference %.2e\n",
                  info.ksp_type.c_str(), info.pc_type.c_str(), nsp ? "on" : "off", sp.iterations,
                  sr.iterations, sp.true_rel_residual, du);
    CHECK(sp.converged);
    CHECK(info.pc_type == "gamg" && info.near_nullspace == nsp);
    CHECK(du < 1e-8);
  }
}

// Patch tests with PETSc CG + GAMG: exact linear fields, both plane conditions.
double patch_error(PlaneCondition plane, const Bcs& bcs,
                   const std::function<std::array<double, 2>(double, double)>& exact, int& its) {
  Problem p = build(data("kirsch_L0.msh"), plane, bcs);
  std::vector<double> u;
  const SolverStats st = petsc_solve(p, u, {true, ""});
  CHECK(st.converged);
  its = st.iterations;
  double err = 0.0, ref = 0.0;
  for (int i = 0; i < p.dm.num_owned; ++i) {
    const auto ue = exact(p.dm.local.xy[2 * i], p.dm.local.xy[2 * i + 1]);
    for (int c = 0; c < 2; ++c) {
      err = std::max(err, std::abs(u[2 * i + c] - ue[c]));
      ref = std::max(ref, std::abs(ue[c]));
    }
  }
  return max_all(err) / max_all(ref);
}

void test_patch_gamg() {
  for (PlaneCondition plane : {PlaneCondition::Stress, PlaneCondition::Strain}) {
    const char* pname = plane == PlaneCondition::Stress ? "plane stress" : "plane strain";
    int it = 0;
    const double a[3] = {1e-4, 2e-3, -1e-3}, bb[3] = {-5e-4, 1e-3, 3e-3};
    auto lin = [&](double x, double y) {
      return std::array<double, 2>{a[0] + a[1] * x + a[2] * y, bb[0] + bb[1] * x + bb[2] * y};
    };
    const double ed = patch_error(
        plane,
        [&](const Mesh& m, DirichletDofs& d, std::vector<double>&) {
          for (const char* g : {"hole", "left_sym", "bottom_sym", "right", "top"}) {
            add_dirichlet_group(m, g, 0, [&](double x, double y) { return lin(x, y)[0]; }, d);
            add_dirichlet_group(m, g, 1, [&](double x, double y) { return lin(x, y)[1]; }, d);
          }
        },
        lin, it);
    if (g_rank == 0)
      std::printf("  Dirichlet patch, %s: %d iterations, max rel. error %.2e\n", pname, it, ed);
    CHECK(ed < 1e-9);

    const bool stress = plane == PlaneCondition::Stress;
    const double Ep = stress ? kE : kE / (1 - kNu * kNu), nup = stress ? kNu : kNu / (1 - kNu);
    for (double syy : {0.0, -40e6}) {
      const double sxx = 100e6;
      const double exx = (sxx - nup * syy) / Ep, eyy = (syy - nup * sxx) / Ep;
      const double en = patch_error(
          plane,
          [&](const Mesh& m, DirichletDofs& d, std::vector<double>& f) {
            add_dirichlet_group(m, "left_sym", 0, 0.0, d);
            add_dirichlet_group(m, "bottom_sym", 1, 0.0, d);
            for (const char* g : {"right", "top", "hole"})
              add_traction(m, g, kT,
                           [&](double, double, double nx, double ny) {
                             return std::array<double, 2>{sxx * nx, syy * ny};
                           },
                           f);
          },
          [&](double x, double y) { return std::array<double, 2>{exx * x, eyy * y}; }, it);
      if (g_rank == 0)
        std::printf("  Neumann patch (%s), %s: %d iterations, max rel. error %.2e\n",
                    syy == 0.0 ? "uniaxial" : "biaxial", pname, it, en);
      CHECK(en < 1e-9);
    }
  }
}

// Kirsch error metrics from the PETSc (GAMG) solution equal the in-house ones.
void test_error_metrics() {
  for (const char* mesh : {"kirsch_L0.msh", "kirsch_L1.msh"}) {
    Problem p = kirsch_problem(mesh);
    std::vector<double> u_ref, u;
    CHECK(solve_distributed(p.A, p.b, u_ref, kOpt, MPI_COMM_WORLD).converged);
    CHECK(petsc_solve(p, u, {true, ""}).converged);
    const KirschMetrics mr = metrics(p, u_ref), mp = metrics(p, u);
    const double dd = rel(mp.l2_rel_displacement_error, mr.l2_rel_displacement_error);
    const double de = rel(mp.energy_rel_error, mr.energy_rel_error);
    const double ds = rel(mp.l2_rel_stress_error, mr.l2_rel_stress_error);
    const double dscf = rel(mp.scf_numerical, mr.scf_numerical);
    if (g_rank == 0)
      std::printf("  %s: displacement L2 %.8e vs %.8e (%.1e), energy %.8e vs %.8e (%.1e), "
                  "stress L2 %.8e vs %.8e (%.1e), SCF %.8f vs %.8f (%.1e)\n",
                  mesh, mp.l2_rel_displacement_error, mr.l2_rel_displacement_error, dd,
                  mp.energy_rel_error, mr.energy_rel_error, de, mp.l2_rel_stress_error,
                  mr.l2_rel_stress_error, ds, mp.scf_numerical, mr.scf_numerical, dscf);
    CHECK(dd < 1e-8 && de < 1e-8 && ds < 1e-8 && dscf < 1e-8);
    CHECK(mp.scf_x == mr.scf_x && mp.scf_y == mr.scf_y);
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);
  petsc_initialize();
  if (g_rank == 0) std::printf("test_petsc on %d rank(s)\n", g_size);

  const std::pair<const char*, void (*)()> tests[] = {
      {"test_cross_check_jacobi", test_cross_check_jacobi},
      {"test_gamg_vs_inhouse", test_gamg_vs_inhouse},
      {"test_patch_gamg", test_patch_gamg},
      {"test_error_metrics", test_error_metrics}};
  for (const auto& [name, fn] : tests) {
    const int before = g_failures;
    fn();
    int mine = g_failures - before, fails = 0;
    MPI_Allreduce(&mine, &fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) std::printf("[%s] %s\n", fails == 0 ? "PASS" : "FAIL", name);
  }
  int fails = 0;
  MPI_Allreduce(&g_failures, &fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) std::printf("%d failures\n", fails);
  petsc_finalize();
  MPI_Finalize();
  return fails == 0 ? 0 : 1;
}
