// Patch tests on kirsch_L0: linear displacement fields must be reproduced exactly.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "cstfem/assembly.hpp"
#include "cstfem/bc.hpp"
#include "cstfem/solver.hpp"
#include "test_assert.hpp"

using namespace cstfem;

namespace {

const double kE = 210e9, kNu = 0.3, kT = 0.01;

Mesh load_l0() { return read_gmsh(std::string(CSTFEM_TEST_DATA_DIR) + "/kirsch_L0.msh"); }

const char* name(PlaneCondition p) {
  return p == PlaneCondition::Stress ? "plane stress" : "plane strain";
}

// Assembles K, applies Dirichlet BCs to (K, f), solves with cg_jacobi (rtol 1e-12).
std::vector<double> solve(const Mesh& m, PlaneCondition plane, const DirichletDofs& dofs,
                          std::vector<double> f, SolverStats& st) {
  CsrMatrix K = build_stiffness_pattern(m);
  assemble_stiffness(m, constitutive_matrix(kE, kNu, plane), kT, K);
  apply_dirichlet(dofs, K, f);
  CgJacobiSolver solver({1e-12, 20000, false});
  solver.setup(K);
  std::vector<double> u;
  st = solver.solve(f, u);
  return u;
}

// max_n |u - u_exact| / max_n |u_exact| over all DOFs.
template <class Exact>
double rel_error(const Mesh& m, const std::vector<double>& u, Exact exact) {
  double err = 0.0, ref = 0.0;
  for (std::size_t n = 0; n < m.num_nodes(); ++n) {
    const auto ue = exact(m.xy[2 * n], m.xy[2 * n + 1]);
    for (int c = 0; c < 2; ++c) {
      err = std::max(err, std::abs(u[2 * n + c] - ue[c]));
      ref = std::max(ref, std::abs(ue[c]));
    }
  }
  return err / ref;
}

void test_dirichlet_patch() {
  const Mesh m = load_l0();
  // u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y on every boundary group.
  const double a[3] = {1e-4, 2e-3, -1e-3}, b[3] = {-5e-4, 1e-3, 3e-3};
  auto exact = [&](double x, double y) {
    return std::array<double, 2>{a[0] + a[1] * x + a[2] * y, b[0] + b[1] * x + b[2] * y};
  };
  for (PlaneCondition plane : {PlaneCondition::Stress, PlaneCondition::Strain}) {
    DirichletDofs dofs;
    for (const char* g : {"hole", "left_sym", "bottom_sym", "right", "top"}) {
      add_dirichlet_group(m, g, 0, [&](double x, double y) { return exact(x, y)[0]; }, dofs);
      add_dirichlet_group(m, g, 1, [&](double x, double y) { return exact(x, y)[1]; }, dofs);
    }
    SolverStats st;
    const auto u = solve(m, plane, dofs, std::vector<double>(2 * m.num_nodes(), 0.0), st);
    const double err = rel_error(m, u, exact);
    std::printf("  Dirichlet patch, %s: %zu fixed DOFs, %d PCG iterations, max rel. error %.2e\n",
                name(plane), dofs.size(), st.iterations, err);
    CHECK(st.converged);
    CHECK(err < 1e-9);
  }
}

// Uniform stress (sxx, syy, 0) with symmetry BCs and traction sigma.n on right, top and hole.
// Exact solution (txy = 0):
//   exx = (sxx - nu' syy) / E',  eyy = (syy - nu' sxx) / E',
//   ux = exx x,  uy = eyy y   (zero on x = 0 and y = 0, matching the symmetry BCs),
// with E' = E, nu' = nu (plane stress) and E' = E / (1 - nu^2), nu' = nu / (1 - nu)
// (plane strain). For the uniaxial case syy = 0:
//   ux = sigma0 x / E',  uy = -nu' sigma0 y / E'.
// `flip_group` negates the normal passed to the traction of that group (to check the test
// is sensitive to the normal orientation); empty for the normal case.
double neumann_patch(const Mesh& m, PlaneCondition plane, double sxx, double syy,
                     const std::string& flip_group, SolverStats& st) {
  const bool stress = plane == PlaneCondition::Stress;
  const double Ep = stress ? kE : kE / (1 - kNu * kNu);
  const double nup = stress ? kNu : kNu / (1 - kNu);
  const double exx = (sxx - nup * syy) / Ep, eyy = (syy - nup * sxx) / Ep;

  DirichletDofs dofs;
  add_dirichlet(m, parse_dirichlet("left_sym:ux=0 ; bottom_sym:uy=0"), dofs);
  std::vector<double> f(2 * m.num_nodes(), 0.0);
  for (const char* g : {"right", "top", "hole"}) {
    const double s = (flip_group == g) ? -1.0 : 1.0;
    add_traction(m, g, kT,
                 [&](double, double, double nx, double ny) {
                   return std::array<double, 2>{sxx * s * nx, syy * s * ny};
                 },
                 f);
  }
  const auto u = solve(m, plane, dofs, f, st);
  return rel_error(m, u, [&](double x, double y) { return std::array<double, 2>{exx * x, eyy * y}; });
}

void test_neumann_patch() {
  const Mesh m = load_l0();
  const double s0 = 100e6;
  for (PlaneCondition plane : {PlaneCondition::Stress, PlaneCondition::Strain}) {
    SolverStats st;
    const double err = neumann_patch(m, plane, s0, 0.0, "", st);
    std::printf("  Neumann patch (uniaxial), %s: %d PCG iterations, max rel. error %.2e\n",
                name(plane), st.iterations, err);
    CHECK(st.converged);
    CHECK(err < 1e-9);

    // Biaxial variant: makes the `top` traction non-zero too.
    const double err2 = neumann_patch(m, plane, s0, -0.4 * s0, "", st);
    std::printf("  Neumann patch (biaxial),  %s: %d PCG iterations, max rel. error %.2e\n",
                name(plane), st.iterations, err2);
    CHECK(st.converged);
    CHECK(err2 < 1e-9);
  }
}

void test_neumann_patch_detects_flipped_normal() {
  // With the normal of any one loaded group reversed, the patch test must fail clearly.
  const Mesh m = load_l0();
  for (const char* g : {"right", "top", "hole"}) {
    SolverStats st;
    const double err = neumann_patch(m, PlaneCondition::Stress, 100e6, -40e6, g, st);
    std::printf("  flipped normal on %-5s -> max rel. error %.2e\n", g, err);
    CHECK(err > 1e-2);
  }
}

}  // namespace

int main() {
  RUN_TEST(test_dirichlet_patch);
  RUN_TEST(test_neumann_patch);
  RUN_TEST(test_neumann_patch_detects_flipped_normal);
  return test_summary();
}
