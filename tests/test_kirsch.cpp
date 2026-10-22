// Kirsch analytical field: SCF, hole boundary, far field, polar/Cartesian consistency.

#include <cmath>
#include <cstdio>
#include <random>

#include "cstfem/kirsch.hpp"
#include "test_assert.hpp"

using namespace cstfem;

namespace {

const KirschParams kP{100e6, 0.1};
const double kPi = 3.14159265358979323846;

void test_scf() {
  // sigma_tt(r = a, theta = 90 deg) = 3 sigma_inf, and sigma_xx at (0, a) is the same value.
  CHECK_NEAR(kirsch_polar(kP, kP.a, kPi / 2).tt / kP.sigma_inf, 3.0, 1e-14);
  CHECK_NEAR(kirsch_stress(kP, 0.0, kP.a)[0] / kP.sigma_inf, 3.0, 1e-14);
  // At theta = 0 on the hole: sigma_tt = -sigma_inf.
  CHECK_NEAR(kirsch_polar(kP, kP.a, 0.0).tt / kP.sigma_inf, -1.0, 1e-14);
}

void test_hole_traction_free() {
  for (int k = 0; k <= 90; ++k) {
    const PolarStress s = kirsch_polar(kP, kP.a, k * kPi / 180.0);
    CHECK_NEAR(s.rr / kP.sigma_inf, 0.0, 1e-14);
    CHECK_NEAR(s.rt / kP.sigma_inf, 0.0, 1e-14);
  }
  // The traction callback with n = -(x, y)/a on the hole is zero.
  const TractionFn t = kirsch_traction(kP);
  for (int k = 0; k <= 90; k += 5) {
    const double th = k * kPi / 180.0, x = kP.a * std::cos(th), y = kP.a * std::sin(th);
    const auto tr = t(x, y, -std::cos(th), -std::sin(th));
    CHECK_NEAR(tr[0] / kP.sigma_inf, 0.0, 1e-14);
    CHECK_NEAR(tr[1] / kP.sigma_inf, 0.0, 1e-14);
  }
}

void test_far_field() {
  // Perturbation decays like (a/r)^2; at r = 1e4 a it is O(1e-8).
  for (int k = 0; k <= 8; ++k) {
    const double th = k * kPi / 16.0, r = 1e4 * kP.a;
    const Vec3 s = kirsch_stress(kP, r * std::cos(th), r * std::sin(th));
    CHECK_NEAR(s[0] / kP.sigma_inf, 1.0, 1e-7);
    CHECK_NEAR(s[1] / kP.sigma_inf, 0.0, 1e-7);
    CHECK_NEAR(s[2] / kP.sigma_inf, 0.0, 1e-7);
  }
  // Traction on x = L with n = (1, 0) tends to (sigma_inf, 0).
  const auto tr = kirsch_traction(kP)(1e4 * kP.a, 0.0, 1.0, 0.0);
  CHECK_NEAR(tr[0] / kP.sigma_inf, 1.0, 1e-7);
  CHECK_NEAR(tr[1] / kP.sigma_inf, 0.0, 1e-7);
}

void test_conversion_consistency() {
  std::mt19937 gen(5);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int i = 0; i < 100; ++i) {
    const PolarStress p{u(gen), u(gen), u(gen)};
    const double th = 3.0 * u(gen);
    const Vec3 c = polar_to_cartesian(p, th);
    const PolarStress back = cartesian_to_polar(c, th);
    CHECK_NEAR(back.rr, p.rr, 1e-14);
    CHECK_NEAR(back.tt, p.tt, 1e-14);
    CHECK_NEAR(back.rt, p.rt, 1e-14);
    // Invariants: trace and determinant of the 2x2 stress tensor.
    CHECK_NEAR(c[0] + c[1], p.rr + p.tt, 1e-14);
    CHECK_NEAR(c[0] * c[1] - c[2] * c[2], p.rr * p.tt - p.rt * p.rt, 1e-14);
  }
  // On the x axis (theta = 0) the polar and Cartesian components coincide.
  const PolarStress p = kirsch_polar(kP, 0.3, 0.0);
  const Vec3 c = kirsch_stress(kP, 0.3, 0.0);
  CHECK_NEAR(c[0], p.rr, 1e-6);
  CHECK_NEAR(c[1], p.tt, 1e-6);
  CHECK_NEAR(c[2], p.rt, 1e-6);
}

}  // namespace

int main() {
  RUN_TEST(test_scf);
  RUN_TEST(test_hole_traction_free);
  RUN_TEST(test_far_field);
  RUN_TEST(test_conversion_consistency);
  return test_summary();
}
