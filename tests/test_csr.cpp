// Global CSR stiffness on kirsch_L0: pattern size, symmetry, rigid-body null space.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <utility>

#include "cstfem/assembly.hpp"
#include "test_assert.hpp"

using namespace cstfem;

namespace {

Mesh load_l0() { return read_gmsh(std::string(CSTFEM_TEST_DATA_DIR) + "/kirsch_L0.msh"); }

double max_abs(const std::vector<double>& v) {
  double m = 0.0;
  for (double x : v) m = std::max(m, std::abs(x));
  return m;
}

// Infinity norm (max absolute row sum).
double norm_inf(const CsrMatrix& K) {
  double m = 0.0;
  for (int i = 0; i < K.num_rows; ++i) {
    double s = 0.0;
    for (int k = K.row_ptr[i]; k < K.row_ptr[i + 1]; ++k) s += std::abs(K.vals[k]);
    m = std::max(m, s);
  }
  return m;
}

void test_pattern() {
  const Mesh m = load_l0();
  const CsrMatrix K = build_stiffness_pattern(m);

  // Predicted nnz: each node couples with itself and with each edge neighbour, 2x2 per pair.
  std::set<std::pair<int, int>> edges;
  for (std::size_t e = 0; e < m.num_triangles(); ++e)
    for (int k = 0; k < 3; ++k) {
      int a = m.triangles[3 * e + k], b = m.triangles[3 * e + (k + 1) % 3];
      edges.insert({std::min(a, b), std::max(a, b)});
    }
  const std::size_t predicted = 4 * (m.num_nodes() + 2 * edges.size());
  std::printf("  kirsch_L0: %zu nodes, %zu edges, %d DOFs, nnz = %zu (predicted %zu)\n",
              m.num_nodes(), edges.size(), K.num_rows, K.nnz(), predicted);
  CHECK(K.nnz() == predicted);
  CHECK(K.num_rows == 2 * static_cast<int>(m.num_nodes()) && K.num_cols == K.num_rows);
  CHECK(K.row_ptr.back() == static_cast<int>(K.nnz()));

  bool sorted = true, has_diag = true;
  for (int i = 0; i < K.num_rows; ++i) {
    for (int k = K.row_ptr[i] + 1; k < K.row_ptr[i + 1]; ++k)
      if (K.cols[k - 1] >= K.cols[k]) sorted = false;
    if (K.find(i, i) < 0) has_diag = false;
  }
  CHECK(sorted);
  CHECK(has_diag);
}

void test_symmetry_and_rigid_modes() {
  const Mesh m = load_l0();
  for (PlaneCondition plane : {PlaneCondition::Stress, PlaneCondition::Strain}) {
    CsrMatrix K = build_stiffness_pattern(m);
    assemble_stiffness(m, constitutive_matrix(210e9, 0.3, plane), 0.01, K);
    const double kmax = max_abs(K.vals);

    int asym = 0;
    for (int i = 0; i < K.num_rows; ++i)
      for (int k = K.row_ptr[i]; k < K.row_ptr[i + 1]; ++k) {
        const int t = K.find(K.cols[k], i);
        if (t < 0 || std::abs(K.vals[k] - K.vals[t]) > 1e-12 * kmax) ++asym;
      }
    CHECK(asym == 0);

    // Rigid-body modes before BCs: ||K r||_inf <= 1e-12 ||K||_inf ||r||_inf.
    const std::size_t nn = m.num_nodes();
    std::vector<double> tx(2 * nn, 0.0), ty(2 * nn, 0.0), rot(2 * nn, 0.0), y;
    for (std::size_t n = 0; n < nn; ++n) {
      tx[2 * n] = 1.0;
      ty[2 * n + 1] = 1.0;
      rot[2 * n] = -m.xy[2 * n + 1];
      rot[2 * n + 1] = m.xy[2 * n];
    }
    const double knorm = norm_inf(K);
    for (const auto* r : {&tx, &ty, &rot}) {
      K.multiply(*r, y);
      const double rel = max_abs(y) / (knorm * max_abs(*r));
      std::printf("  %s: rigid mode %s: ||K r|| / (||K|| ||r||) = %.2e\n",
                  plane == PlaneCondition::Stress ? "plane stress" : "plane strain",
                  r == &tx ? "tx " : r == &ty ? "ty " : "rot", rel);
      CHECK(rel <= 1e-12);
    }
  }
}

}  // namespace

int main() {
  RUN_TEST(test_pattern);
  RUN_TEST(test_symmetry_and_rigid_modes);
  return test_summary();
}
