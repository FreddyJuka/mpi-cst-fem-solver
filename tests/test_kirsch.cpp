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

void test_displacement_vs_sympy() {
  // Reference values printed by python/derive_kirsch_displacement.py
  // (sigma_inf = 100e6, a = 0.1, E = 210e9, nu = 0.3).
  struct Ref {
    PlaneCondition plane;
    double x, y, ux, uy;
  };
  const Ref refs[] = {
      {PlaneCondition::Stress, 0.1, 0.0, 0.00014285714285714285714, 0.0},
      {PlaneCondition::Stress, 0.0, 0.1, 0.0, -0.000047619047619047619048},
      {PlaneCondition::Stress, 0.3, 0.2, 0.00016331360946745562130, -0.000025922795153564384334},
      {PlaneCondition::Stress, 1.0, 1.0, 0.00047941250000000000000, -0.00014298392857142857143},
      {PlaneCondition::Stress, 1.0, 0.0, 0.00048877857142857142857, 0.0},
      {PlaneCondition::Stress, 0.05, 0.9, 0.000023833524260355029586, -0.00013560843144547759932},
      {PlaneCondition::Strain, 0.1, 0.0, 0.00013000000000000000000, 0.0},
      {PlaneCondition::Strain, 0.0, 0.1, 0.0, -0.000043333333333333333333},
      {PlaneCondition::Strain, 0.3, 0.2, 0.00014847844463229078614, -0.000033175542406311637081},
      {PlaneCondition::Strain, 1.0, 1.0, 0.00043612678571428571429, -0.00018541250000000000000},
      {PlaneCondition::Strain, 1.0, 0.0, 0.00044506428571428571429, 0.0},
      {PlaneCondition::Strain, 0.05, 0.9, 0.000021637919864750633981, -0.00017323040946745562130},
  };
  const double scale = 5e-4;  // max |u| among the reference values
  for (const Ref& r : refs) {
    const auto u = kirsch_displacement(kP, 210e9, 0.3, r.plane, r.x, r.y);
    CHECK_NEAR(u[0], r.ux, 1e-14 * scale);
    CHECK_NEAR(u[1], r.uy, 1e-14 * scale);
  }
}

void test_displacement_consistent_with_stress() {
  // Central differences of the displacement, through Hooke's law, give the Kirsch stress.
  for (PlaneCondition plane : {PlaneCondition::Stress, PlaneCondition::Strain}) {
    const double E = 210e9, nu = 0.3, h = 1e-6;
    const Mat3 D = constitutive_matrix(E, nu, plane);
    for (const auto& xy : {std::array<double, 2>{0.15, 0.05}, std::array<double, 2>{0.4, 0.7}}) {
      auto u = [&](double x, double y) { return kirsch_displacement(kP, E, nu, plane, x, y); };
      const auto ux1 = u(xy[0] + h, xy[1]), ux0 = u(xy[0] - h, xy[1]);
      const auto uy1 = u(xy[0], xy[1] + h), uy0 = u(xy[0], xy[1] - h);
      const Vec3 eps = {(ux1[0] - ux0[0]) / (2 * h), (uy1[1] - uy0[1]) / (2 * h),
                        (uy1[0] - uy0[0]) / (2 * h) + (ux1[1] - ux0[1]) / (2 * h)};
      const Vec3 sig = stress_from_strain(D, eps), ref = kirsch_stress(kP, xy[0], xy[1]);
      for (int i = 0; i < 3; ++i) CHECK_NEAR(sig[i] / kP.sigma_inf, ref[i] / kP.sigma_inf, 1e-6);
    }
  }
}

}  // namespace

int main() {
  RUN_TEST(test_scf);
  RUN_TEST(test_hole_traction_free);
  RUN_TEST(test_far_field);
  RUN_TEST(test_conversion_consistency);
  RUN_TEST(test_displacement_vs_sympy);
  RUN_TEST(test_displacement_consistent_with_stress);
  return test_summary();
}
