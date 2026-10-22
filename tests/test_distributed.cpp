// MPI tests of distributed assembly, BCs, halo exchange and Jacobi-PCG against the serial code.
// Run with any number of ranks; rank 0 also runs the serial reference.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "cstfem/assembly.hpp"
#include "cstfem/bc.hpp"
#include "cstfem/distributed.hpp"
#include "cstfem/kirsch.hpp"
#include "cstfem/postprocess.hpp"

using namespace cstfem;

namespace {

int g_rank = 0, g_size = 1, g_failures = 0;
Partitioner g_partitioner = Partitioner::Metis;  // --partitioner=parmetis
const double kE = 210e9, kNu = 0.3, kT = 0.01;
const KirschParams kKirsch{100e6, 0.1};

#define CHECK(cond)                                                                            \
  do {                                                                                         \
    if (!(cond)) {                                                                             \
      std::fprintf(stderr, "[rank %d] %s:%d: CHECK failed: %s\n", g_rank, __FILE__, __LINE__, \
                   #cond);                                                                     \
      ++g_failures;                                                                            \
    }                                                                                          \
  } while (0)

std::string data(const char* name) { return std::string(CSTFEM_TEST_DATA_DIR) + "/" + name; }

template <class T>
std::vector<T> gather_all(const std::vector<T>& mine, MPI_Datatype type) {
  int n = static_cast<int>(mine.size());
  std::vector<int> counts(g_size), displs(g_size);
  MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  int total = 0;
  for (int r = 0; r < g_size; ++r) {
    displs[r] = total;
    total += counts[r];
  }
  std::vector<T> all(g_rank == 0 ? total : 0);
  MPI_Gatherv(mine.data(), n, type, all.data(), counts.data(), displs.data(), type, 0,
              MPI_COMM_WORLD);
  return all;
}

double max_all(double v) {
  double m = 0.0;
  MPI_Allreduce(&v, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return m;
}

// Rank 0: global DOF (distributed numbering) -> serial DOF, via Gmsh tags of owned nodes.
std::vector<int> dof_map_to_serial(const DistributedMesh& dm, const Mesh* full) {
  std::vector<std::int64_t> tags;
  for (int i = 0; i < dm.num_owned; ++i) tags.push_back(static_cast<std::int64_t>(dm.local.node_tags[i]));
  const auto all = gather_all(tags, MPI_INT64_T);  // in global node order (ranks in order)
  std::vector<int> map;
  if (g_rank == 0) {
    std::unordered_map<std::int64_t, int> serial;
    for (std::size_t n = 0; n < full->num_nodes(); ++n)
      serial[static_cast<std::int64_t>(full->node_tags[n])] = static_cast<int>(n);
    map.resize(2 * all.size());
    for (std::size_t g = 0; g < all.size(); ++g) {
      map[2 * g] = 2 * serial.at(all[g]);
      map[2 * g + 1] = 2 * serial.at(all[g]) + 1;
    }
  }
  return map;
}

// Owned vector -> gathered on rank 0 in serial DOF numbering.
std::vector<double> gather_to_serial(const std::vector<double>& owned, const std::vector<int>& map) {
  const auto all = gather_all(owned, MPI_DOUBLE);  // global DOF order
  std::vector<double> out;
  if (g_rank == 0) {
    out.assign(all.size(), 0.0);
    for (std::size_t g = 0; g < all.size(); ++g) out[map[g]] = all[g];
  }
  return out;
}

double rel_max_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0, r = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d = std::max(d, std::abs(a[i] - b[i]));
    r = std::max(r, std::abs(b[i]));
  }
  return d / r;
}

struct Parallel {
  DistributedMesh dm;
  DofLayout layout;
  DistMatrix A;
};

Parallel build(const std::string& path, PlaneCondition plane) {
  Parallel p;
  p.dm = read_partition_distribute(path, MPI_COMM_WORLD, nullptr, g_partitioner);
  p.layout = make_dof_layout(p.dm, MPI_COMM_WORLD);
  p.A = assemble_distributed(p.dm, p.layout, constitutive_matrix(kE, kNu, plane), kT,
                             MPI_COMM_WORLD);
  return p;
}

// Rank 0: gathered distributed matrix in serial numbering, as {(row, col) -> value}.
std::map<std::pair<int, int>, double> gather_matrix(const DistMatrix& A, const std::vector<int>& map) {
  std::vector<int> rc;
  std::vector<double> v;
  for (int i = 0; i < A.n_owned; ++i) {
    for (int k = A.loc.row_ptr[i]; k < A.loc.row_ptr[i + 1]; ++k) {
      rc.insert(rc.end(), {A.first_row + i, A.first_row + A.loc.cols[k]});
      v.push_back(A.loc.vals[k]);
    }
    for (int k = A.ghost.row_ptr[i]; k < A.ghost.row_ptr[i + 1]; ++k) {
      rc.insert(rc.end(), {A.first_row + i, A.halo_globals[A.ghost.cols[k]]});
      v.push_back(A.ghost.vals[k]);
    }
  }
  const auto all_rc = gather_all(rc, MPI_INT);
  const auto all_v = gather_all(v, MPI_DOUBLE);
  std::map<std::pair<int, int>, double> out;
  for (std::size_t i = 0; i < all_v.size(); ++i)
    out[{map[all_rc[2 * i]], map[all_rc[2 * i + 1]]}] = all_v[i];
  return out;
}

void compare_matrix(const std::map<std::pair<int, int>, double>& dist, const CsrMatrix& K,
                    const char* label) {
  if (g_rank != 0) return;
  double kmax = 0.0, diff = 0.0;
  for (double v : K.vals) kmax = std::max(kmax, std::abs(v));
  CHECK(dist.size() == K.nnz());
  for (int i = 0; i < K.num_rows; ++i)
    for (int k = K.row_ptr[i]; k < K.row_ptr[i + 1]; ++k) {
      const auto it = dist.find({i, K.cols[k]});
      if (it == dist.end()) {
        CHECK(it != dist.end());
        continue;
      }
      diff = std::max(diff, std::abs(it->second - K.vals[k]));
    }
  std::printf("  %s: %zu entries, max |K_par - K_ser| / max |K| = %.2e\n", label, dist.size(),
              diff / kmax);
  CHECK(diff <= 1e-14 * kmax);
}

// ---- tests ----------------------------------------------------------------------------------

void test_spmv() {
  Mesh full;
  if (g_rank == 0) full = read_gmsh(data("kirsch_L1.msh"));
  Parallel p = build(data("kirsch_L1.msh"), PlaneCondition::Stress);
  const auto map = dof_map_to_serial(p.dm, &full);
  auto f = [](std::size_t tag, int c) { return std::sin(0.7 * static_cast<double>(tag) + 1.3 * c) + 0.1 * c; };

  std::vector<double> x(p.A.n_owned), y;
  for (int i = 0; i < p.dm.num_owned; ++i)
    for (int c = 0; c < 2; ++c) x[2 * i + c] = f(p.dm.local.node_tags[i], c);
  DistOperator(p.A).apply(x, y);
  const auto y_par = gather_to_serial(y, map);
  if (g_rank == 0) {
    CsrMatrix K = build_stiffness_pattern(full);
    assemble_stiffness(full, constitutive_matrix(kE, kNu, PlaneCondition::Stress), kT, K);
    std::vector<double> xs(2 * full.num_nodes()), ys;
    for (std::size_t n = 0; n < full.num_nodes(); ++n)
      for (int c = 0; c < 2; ++c) xs[2 * n + c] = f(full.node_tags[n], c);
    K.multiply(xs, ys);
    const double d = rel_max_diff(y_par, ys);
    std::printf("  SpMV kirsch_L1: max rel. difference vs serial %.2e\n", d);
    CHECK(d < 1e-14);
  }
}

void test_matrix_and_bcs() {
  Mesh full;
  if (g_rank == 0) full = read_gmsh(data("kirsch_L0.msh"));
  Parallel p = build(data("kirsch_L0.msh"), PlaneCondition::Strain);
  const auto map = dof_map_to_serial(p.dm, &full);

  CsrMatrix K;
  if (g_rank == 0) {
    K = build_stiffness_pattern(full);
    assemble_stiffness(full, constitutive_matrix(kE, kNu, PlaneCondition::Strain), kT, K);
  }
  compare_matrix(gather_matrix(p.A, map), K, "matrix before BCs");

  // Non-zero Dirichlet values on three groups and Kirsch tractions, so that both the matrix
  // elimination and the right-hand-side corrections are exercised.
  auto ux = [](double x, double y) { return 1e-4 * (1 + 2 * x - y); };
  auto uy = [](double x, double y) { return 1e-4 * (0.5 - x + 3 * y); };
  auto bcs = [&](const Mesh& m, DirichletDofs& d, std::vector<double>& f) {
    add_dirichlet_group(m, "left_sym", 0, ux, d);
    add_dirichlet_group(m, "bottom_sym", 1, uy, d);
    add_dirichlet_group(m, "hole", 0, ux, d);
    add_dirichlet_group(m, "hole", 1, uy, d);
    add_traction(m, "right", kT, kirsch_traction(kKirsch), f);
    add_traction(m, "top", kT, kirsch_traction(kKirsch), f);
  };
  DirichletDofs local;
  std::vector<double> f_local(p.layout.n_local, 0.0);
  bcs(p.dm.local, local, f_local);
  std::vector<double> b = accumulate_owned(p.layout, f_local);
  apply_dirichlet_distributed(p.dm, p.layout, local, p.A, b, MPI_COMM_WORLD);
  const auto dist_after = gather_matrix(p.A, map);
  const auto b_par = gather_to_serial(b, map);
  if (g_rank == 0) {
    DirichletDofs d;
    std::vector<double> bs(2 * full.num_nodes(), 0.0);
    bcs(full, d, bs);
    apply_dirichlet(d, K, bs);
    const double db = rel_max_diff(b_par, bs);
    std::printf("  rhs after BCs: max rel. difference vs serial %.2e\n", db);
    CHECK(db < 1e-14);
  }
  compare_matrix(dist_after, K, "matrix after BCs");
}

// Parallel solve with BCs set by `bcs(local mesh, dirichlet, f_local)`; returns the max
// relative nodal error against `exact`.
double parallel_patch(PlaneCondition plane,
                      const std::function<void(const Mesh&, DirichletDofs&, std::vector<double>&)>& bcs,
                      const std::function<std::array<double, 2>(double, double)>& exact,
                      int& iterations) {
  Parallel p = build(data("kirsch_L0.msh"), plane);
  DirichletDofs local;
  std::vector<double> f_local(p.layout.n_local, 0.0);
  bcs(p.dm.local, local, f_local);
  std::vector<double> b = accumulate_owned(p.layout, f_local);
  apply_dirichlet_distributed(p.dm, p.layout, local, p.A, b, MPI_COMM_WORLD);
  std::vector<double> u;
  const SolverStats st = solve_distributed(p.A, b, u, {1e-12, 20000, false}, MPI_COMM_WORLD);
  CHECK(st.converged);
  iterations = st.iterations;
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

void test_patch() {
  for (PlaneCondition plane : {PlaneCondition::Stress, PlaneCondition::Strain}) {
    const char* pname = plane == PlaneCondition::Stress ? "plane stress" : "plane strain";
    int it = 0;
    // Dirichlet patch: linear field on every boundary group.
    const double a[3] = {1e-4, 2e-3, -1e-3}, bb[3] = {-5e-4, 1e-3, 3e-3};
    auto lin = [&](double x, double y) {
      return std::array<double, 2>{a[0] + a[1] * x + a[2] * y, bb[0] + bb[1] * x + bb[2] * y};
    };
    const double ed = parallel_patch(
        plane,
        [&](const Mesh& m, DirichletDofs& d, std::vector<double>&) {
          for (const char* g : {"hole", "left_sym", "bottom_sym", "right", "top"}) {
            add_dirichlet_group(m, g, 0, [&](double x, double y) { return lin(x, y)[0]; }, d);
            add_dirichlet_group(m, g, 1, [&](double x, double y) { return lin(x, y)[1]; }, d);
          }
        },
        lin, it);
    if (g_rank == 0) std::printf("  Dirichlet patch, %s: %d iterations, max rel. error %.2e\n", pname, it, ed);
    CHECK(ed < 1e-9);

    // Neumann patch: uniform (sxx, syy), symmetry BCs, traction on right, top and hole.
    const bool stress = plane == PlaneCondition::Stress;
    const double Ep = stress ? kE : kE / (1 - kNu * kNu), nup = stress ? kNu : kNu / (1 - kNu);
    for (double syy : {0.0, -40e6}) {
      const double sxx = 100e6;
      const double exx = (sxx - nup * syy) / Ep, eyy = (syy - nup * sxx) / Ep;
      const double en = parallel_patch(
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

void test_serial_equivalence() {
  for (const char* name : {"kirsch_L0.msh", "kirsch_L1.msh"}) {
    const SolverOptions opt{1e-12, 20000, false};
    const Mat3 D = constitutive_matrix(kE, kNu, PlaneCondition::Stress);
    auto bcs = [&](const Mesh& m, DirichletDofs& d, std::vector<double>& f) {
      add_dirichlet(m, parse_dirichlet("left_sym:ux=0 ; bottom_sym:uy=0"), d);
      add_traction(m, "right", kT, kirsch_traction(kKirsch), f);
      add_traction(m, "top", kT, kirsch_traction(kKirsch), f);
    };

    Mesh full;
    if (g_rank == 0) full = read_gmsh(data(name));
    Parallel p = build(data(name), PlaneCondition::Stress);
    const auto map = dof_map_to_serial(p.dm, &full);
    DirichletDofs local;
    std::vector<double> f_local(p.layout.n_local, 0.0);
    bcs(p.dm.local, local, f_local);
    std::vector<double> b = accumulate_owned(p.layout, f_local);
    apply_dirichlet_distributed(p.dm, p.layout, local, p.A, b, MPI_COMM_WORLD);
    std::vector<double> u;
    const SolverStats st = solve_distributed(p.A, b, u, opt, MPI_COMM_WORLD);
    CHECK(st.converged);
    const auto u_par = gather_to_serial(u, map);
    const ElasticMaterial mat{kE, kNu, PlaneCondition::Stress};
    const std::vector<double> u_loc = to_mesh_local(p.layout, u);
    const ElementStresses es = compute_element_stresses(p.dm.local, u_loc, D, kNu, mat.plane);
    const KirschMetrics mp =
        kirsch_metrics_distributed(p.dm.local, u_loc, es, kKirsch, mat, MPI_COMM_WORLD);

    if (g_rank == 0) {
      CsrMatrix K = build_stiffness_pattern(full);
      assemble_stiffness(full, D, kT, K);
      DirichletDofs d;
      std::vector<double> f(2 * full.num_nodes(), 0.0), us;
      bcs(full, d, f);
      apply_dirichlet(d, K, f);
      CgJacobiSolver solver(opt);
      solver.setup(K);
      const SolverStats ss = solver.solve(f, us);
      const KirschMetrics ms = kirsch_metrics(
          full, us, compute_element_stresses(full, us, D, kNu, mat.plane), kKirsch, mat);
      const double du = rel_max_diff(u_par, us);
      const double dl2 = std::abs(mp.l2_rel_stress_error / ms.l2_rel_stress_error - 1);
      const double dscf = std::abs(mp.scf_numerical / ms.scf_numerical - 1);
      std::printf("  %s: iterations serial %d / parallel %d; max rel. displacement diff %.2e; "
                  "L2 %.6e vs %.6e (diff %.1e); SCF %.6f vs %.6f (diff %.1e)\n",
                  name, ss.iterations, st.iterations, du, ms.l2_rel_stress_error,
                  mp.l2_rel_stress_error, dl2, ms.scf_numerical, mp.scf_numerical, dscf);
      CHECK(du < 1e-8);
      CHECK(dl2 < 1e-8);
      CHECK(dscf < 1e-8);
      const double dd = std::abs(mp.l2_rel_displacement_error / ms.l2_rel_displacement_error - 1);
      const double de = std::abs(mp.energy_rel_error / ms.energy_rel_error - 1);
      std::printf("    displacement L2 %.6e vs %.6e (diff %.1e); energy %.6e vs %.6e (diff %.1e)\n",
                  ms.l2_rel_displacement_error, mp.l2_rel_displacement_error, dd,
                  ms.energy_rel_error, mp.energy_rel_error, de);
      CHECK(dd < 1e-8);
      CHECK(de < 1e-8);
      CHECK(mp.scf_x == ms.scf_x && mp.scf_y == ms.scf_y);
    }
  }
}

// Solver instrumentation: components sum to the measured solve time (within 5 % on every
// rank), counters are consistent, and the solution is bit-identical with timing on and off.
void test_solve_breakdown() {
  Parallel p = build(data("kirsch_L1.msh"), PlaneCondition::Stress);
  DirichletDofs local;
  std::vector<double> f_local(p.layout.n_local, 0.0);
  add_dirichlet(p.dm.local, parse_dirichlet("left_sym:ux=0 ; bottom_sym:uy=0"), local);
  add_traction(p.dm.local, "right", kT, kirsch_traction(kKirsch), f_local);
  add_traction(p.dm.local, "top", kT, kirsch_traction(kKirsch), f_local);
  std::vector<double> b = accumulate_owned(p.layout, f_local);
  apply_dirichlet_distributed(p.dm, p.layout, local, p.A, b, MPI_COMM_WORLD);
  const SolverOptions opt{1e-10, 20000, false};

  std::vector<double> u_off, u_on;
  const SolverStats s_off = solve_distributed(p.A, b, u_off, opt, MPI_COMM_WORLD);
  SolveBreakdown bd;
  const SolverStats s_on = solve_distributed(p.A, b, u_on, opt, MPI_COMM_WORLD, &bd);

  CHECK(s_off.iterations == s_on.iterations);
  CHECK(u_off.size() == u_on.size() &&
        std::equal(u_off.begin(), u_off.end(), u_on.begin()));  // bit-identical
  const double frac = bd.accounted() / bd.total;
  CHECK(bd.total > 0.0);
  CHECK(std::abs(frac - 1.0) <= 0.05);
  CHECK(bd.spmv_calls == s_on.iterations + 1);  // one per iteration + the true residual
  CHECK(bd.halo_bytes_sent == bd.spmv_calls * static_cast<long long>(
                                  p.A.halo->forward_send_values() * sizeof(double)));
  if (g_size == 1) CHECK(bd.halo_bytes_sent == 0 && bd.allreduce >= 0.0);

  double fmin = 0.0, fmax = 0.0;
  MPI_Reduce(&frac, &fmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&frac, &fmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (g_rank == 0)
    std::printf("  breakdown: %d iterations, accounted fraction over ranks %.4f .. %.4f, "
                "solution identical with timing on/off\n",
                s_on.iterations, fmin, fmax);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--partitioner=parmetis") g_partitioner = Partitioner::ParMetis;
  if (g_rank == 0)
    std::printf("test_distributed on %d rank(s), partitioner %s\n", g_size,
                partitioner_name(g_partitioner).c_str());

  const std::pair<const char*, void (*)()> tests[] = {{"test_spmv", test_spmv},
                                                      {"test_matrix_and_bcs", test_matrix_and_bcs},
                                                      {"test_patch", test_patch},
                                                      {"test_serial_equivalence", test_serial_equivalence},
                                                      {"test_solve_breakdown", test_solve_breakdown}};
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
  MPI_Finalize();
  return fails == 0 ? 0 : 1;
}
