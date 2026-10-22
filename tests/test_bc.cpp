// Boundary conditions: parsing, conflicts, symmetric elimination, outward normals, equilibrium.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "cstfem/assembly.hpp"
#include "cstfem/bc.hpp"
#include "test_assert.hpp"

using namespace cstfem;

namespace {

Mesh load_l0() { return read_gmsh(std::string(CSTFEM_TEST_DATA_DIR) + "/kirsch_L0.msh"); }

void test_parse_dirichlet() {
  const auto s = parse_dirichlet("left_sym:ux=0 ; bottom_sym:uy=0");
  CHECK(s.size() == 2);
  CHECK(s[0].group == "left_sym" && s[0].has_ux && !s[0].has_uy && s[0].ux == 0.0);
  CHECK(s[1].group == "bottom_sym" && !s[1].has_ux && s[1].has_uy && s[1].uy == 0.0);
  const auto both = parse_dirichlet("  clamp : ux = 1e-3 , uy = -2.5 ");
  CHECK(both.size() == 1 && both[0].group == "clamp" && both[0].has_ux && both[0].has_uy &&
        both[0].ux == 1e-3 && both[0].uy == -2.5);
  CHECK_THROWS(parse_dirichlet("g:uz=0"), "unknown component 'uz'");
  CHECK_THROWS(parse_dirichlet("g:ux=abc"), "'abc' is not a number");
  CHECK_THROWS(parse_dirichlet("g ux=0"), "expected 'group:key=value'");
  CHECK_THROWS(parse_dirichlet("g:ux=1,ux=2"), "'ux' given twice");
}

void test_parse_traction() {
  const auto s = parse_traction("right:tx=100e6,ty=0 ; top:ty=-5");
  CHECK(s.size() == 2);
  CHECK(s[0].group == "right" && s[0].tx == 100e6 && s[0].ty == 0.0);
  CHECK(s[1].group == "top" && s[1].tx == 0.0 && s[1].ty == -5.0);
  CHECK_THROWS(parse_traction("right:fx=1"), "unknown component 'fx'");
  CHECK_THROWS(parse_traction("right:tx=1,tx=2"), "'tx' given twice");
}

void test_dirichlet_conflicts() {
  DirichletDofs d;
  d.add(4, 1.0);
  d.add(4, 1.0);  // identical value: fine
  CHECK(d.size() == 1);
  CHECK_THROWS(d.add(4, 2.0), "conflicting Dirichlet values for node 2 (0-based) component ux");

  // `right` and `bottom_sym` share the corner node (L, 0).
  const Mesh m = load_l0();
  DirichletDofs ok;
  add_dirichlet(m, parse_dirichlet("right:ux=1 ; bottom_sym:ux=1,uy=0"), ok);
  DirichletDofs bad;
  CHECK_THROWS(add_dirichlet(m, parse_dirichlet("right:ux=1 ; bottom_sym:ux=0"), bad),
               "conflicting Dirichlet values");
  DirichletDofs unknown;
  CHECK_THROWS(add_dirichlet(m, parse_dirichlet("nowhere:ux=0"), unknown),
               "unknown boundary group 'nowhere'");
}

void test_symmetric_elimination() {
  // K = [4 1 0; 1 3 1; 0 1 2], b = [1 1 1], u_1 = 2:
  // b -> [1 - 1*2, 3*2, 1 - 1*2] = [-1, 6, -1], K -> diag(4, 3, 2).
  CsrMatrix K;
  K.num_rows = K.num_cols = 3;
  K.row_ptr = {0, 2, 5, 7};
  K.cols = {0, 1, 0, 1, 2, 1, 2};
  K.vals = {4, 1, 1, 3, 1, 1, 2};
  std::vector<double> b = {1, 1, 1};
  DirichletDofs d;
  d.add(1, 2.0);
  apply_dirichlet(d, K, b);
  CHECK(b[0] == -1.0 && b[1] == 6.0 && b[2] == -1.0);
  const std::vector<double> expected = {4, 0, 0, 3, 0, 0, 2};
  CHECK(K.vals == expected);
}

void test_outward_normals() {
  // Straight groups: exact normal. Hole: the chord normal must point towards the origin
  // (out of the plate); it deviates from the radial direction by at most half the angle an
  // edge subtends (~5.6 deg on kirsch_L0), so 1 + n.(x, y)/r < 0.01.
  const Mesh m = load_l0();
  struct Expect {
    const char* group;
    double nx, ny;
  };
  for (const Expect& ex : {Expect{"right", 1, 0}, Expect{"top", 0, 1}, Expect{"left_sym", -1, 0},
                           Expect{"bottom_sym", 0, -1}, Expect{"hole", 0, 0}}) {
    const bool hole = std::string(ex.group) == "hole";
    std::vector<double> f(2 * m.num_nodes(), 0.0);
    double worst = 0.0;
    add_traction(m, ex.group, 1.0,
                 [&](double x, double y, double nx, double ny) {
                   const double d = hole ? 1.0 + (nx * x + ny * y) / std::hypot(x, y)
                                         : std::abs(nx - ex.nx) + std::abs(ny - ex.ny);
                   worst = std::max(worst, d);
                   return std::array<double, 2>{0.0, 0.0};
                 },
                 f);
    std::printf("  %-10s max normal deviation %.2e\n", ex.group, worst);
    CHECK(worst < (hole ? 1e-2 : 1e-12));
  }
}

void test_equilibrium() {
  const Mesh m = load_l0();
  const double t = 0.01, tx = 3.0e6, ty = -2.0e6;
  for (const char* g : {"hole", "left_sym", "bottom_sym", "right", "top"}) {
    double length = 0.0;
    for (const auto& e : m.boundary_edges.at(g))
      length += std::hypot(m.xy[2 * e[1]] - m.xy[2 * e[0]], m.xy[2 * e[1] + 1] - m.xy[2 * e[0] + 1]);
    std::vector<double> f(2 * m.num_nodes(), 0.0);
    add_traction(m, g, t, tx, ty, f);
    double fx = 0.0, fy = 0.0;
    for (std::size_t n = 0; n < m.num_nodes(); ++n) {
      fx += f[2 * n];
      fy += f[2 * n + 1];
    }
    const double ex = t * tx * length, ey = t * ty * length;
    std::printf("  %-10s sum f = (%.6e, %.6e), integral = (%.6e, %.6e)\n", g, fx, fy, ex, ey);
    CHECK(std::abs(fx - ex) <= 1e-12 * std::abs(ex));
    CHECK(std::abs(fy - ey) <= 1e-12 * std::abs(ey));
  }
}

}  // namespace

int main() {
  RUN_TEST(test_parse_dirichlet);
  RUN_TEST(test_parse_traction);
  RUN_TEST(test_dirichlet_conflicts);
  RUN_TEST(test_symmetric_elimination);
  RUN_TEST(test_outward_normals);
  RUN_TEST(test_equilibrium);
  return test_summary();
}
