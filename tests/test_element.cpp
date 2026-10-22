// Unit tests for the material law and the CST element.

#include <algorithm>
#include <cmath>
#include <random>

#include "cstfem/element.hpp"
#include "test_assert.hpp"

using namespace cstfem;

namespace {

double max_abs(const Mat6& K) {
  double m = 0.0;
  for (double v : K) m = std::max(m, std::abs(v));
  return m;
}

Vec6 mat_vec(const Mat6& K, const Vec6& u) {
  Vec6 r{};
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) r[i] += K[i * 6 + j] * u[j];
  return r;
}

double dot(const Vec6& a, const Vec6& b) {
  double s = 0.0;
  for (int i = 0; i < 6; ++i) s += a[i] * b[i];
  return s;
}

// Random counter-clockwise triangle with area well away from zero.
TriCoords random_triangle(std::mt19937& gen) {
  std::uniform_real_distribution<double> dist(-2.0, 3.0);
  for (;;) {
    TriCoords xy;
    for (double& v : xy) v = dist(gen);
    const double A = cst_signed_area(xy);
    if (std::abs(A) < 0.5) continue;
    if (A < 0) {
      std::swap(xy[2], xy[4]);
      std::swap(xy[3], xy[5]);
    }
    return xy;
  }
}

// Nodal values of the linear field u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y.
Vec6 linear_field(const TriCoords& xy, const double a[3], const double b[3]) {
  Vec6 u{};
  for (int n = 0; n < 3; ++n) {
    const double x = xy[2 * n], y = xy[2 * n + 1];
    u[2 * n] = a[0] + a[1] * x + a[2] * y;
    u[2 * n + 1] = b[0] + b[1] * x + b[2] * y;
  }
  return u;
}

void test_ke_reference_triangle() {
  // Triangle (0,0), (1,0), (0,1); E = 1, nu = 0, t = 1, plane stress.
  //   A = 1/2, D = diag(1, 1, 1/2).
  //   b = (y2-y3, y3-y1, y1-y2) = (-1, 1, 0),  c = (x3-x2, x1-x3, x2-x1) = (-1, 0, 1),
  //   B = 1/(2A) [b1 0 b2 0 b3 0; 0 c1 0 c2 0 c3; c1 b1 c2 b2 c3 b3]
  //     = [-1  0  1  0  0  0;
  //         0 -1  0  0  0  1;
  //        -1 -1  0  1  1  0]
  //   Ke = t A B^T D B = 1/2 (r1^T r1 + r2^T r2 + 1/2 r3^T r3), r_i = rows of B:
  //   Ke = [ 3/4  1/4 -1/2 -1/4 -1/4    0;
  //          1/4  3/4    0 -1/4 -1/4 -1/2;
  //         -1/2    0  1/2    0    0    0;
  //         -1/4 -1/4    0  1/4  1/4    0;
  //         -1/4 -1/4    0  1/4  1/4    0;
  //            0 -1/2    0    0    0  1/2]
  const Mat6 expected = {0.75,  0.25,  -0.5, -0.25, -0.25, 0.0,   //
                         0.25,  0.75,  0.0,  -0.25, -0.25, -0.5,  //
                         -0.5,  0.0,   0.5,  0.0,   0.0,   0.0,   //
                         -0.25, -0.25, 0.0,  0.25,  0.25,  0.0,   //
                         -0.25, -0.25, 0.0,  0.25,  0.25,  0.0,   //
                         0.0,   -0.5,  0.0,  0.0,   0.0,   0.5};
  const TriCoords xy = {0, 0, 1, 0, 0, 1};
  const Mat6 K = cst_stiffness(xy, constitutive_matrix(1.0, 0.0, PlaneCondition::Stress), 1.0);
  for (int i = 0; i < 36; ++i) CHECK_NEAR(K[i], expected[i], 1e-14);
}

void test_ke_properties() {
  std::mt19937 gen(12345);
  for (PlaneCondition plane : {PlaneCondition::Stress, PlaneCondition::Strain}) {
    for (int trial = 0; trial < 20; ++trial) {
      const TriCoords xy = random_triangle(gen);
      const double E = 210e9, nu = 0.3, t = 0.01;
      const Mat6 K = cst_stiffness(xy, constitutive_matrix(E, nu, plane), t);
      const double kmax = max_abs(K);

      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) CHECK(std::abs(K[i * 6 + j] - K[j * 6 + i]) <= 1e-12 * kmax);

      // Rigid-body modes: translations in x and y, infinitesimal rotation (-y, x).
      Vec6 tx{}, ty{}, rot{};
      double rmax = 0.0;
      for (int n = 0; n < 3; ++n) {
        tx[2 * n] = 1.0;
        ty[2 * n + 1] = 1.0;
        rot[2 * n] = -xy[2 * n + 1];
        rot[2 * n + 1] = xy[2 * n];
        rmax = std::max({rmax, std::abs(xy[2 * n]), std::abs(xy[2 * n + 1])});
      }
      for (const Vec6* r : {&tx, &ty, &rot}) {
        const double scale = (r == &rot ? rmax : 1.0) * kmax;
        for (double v : mat_vec(K, *r)) CHECK(std::abs(v) <= 1e-12 * scale);
      }

      // Constant (non-zero) strain field has positive strain energy.
      const double a[3] = {0.0, 1e-3, 2e-4}, b[3] = {0.0, -1e-4, 5e-4};
      const Vec6 u = linear_field(xy, a, b);
      CHECK(dot(u, mat_vec(K, u)) > 0.0);
    }
  }
}

void test_constitutive_closed_form() {
  const double E = 1.0, nu = 0.3;
  const Mat3 Ds = constitutive_matrix(E, nu, PlaneCondition::Stress);
  // Plane stress: E/(1-nu^2) = 1/0.91.
  const Mat3 Ds_ref = {1.0 / 0.91, 0.3 / 0.91, 0.0, 0.3 / 0.91, 1.0 / 0.91,
                       0.0,        0.0,        0.0, 0.35 / 0.91};
  // Plane strain: E/((1+nu)(1-2nu)) = 1/0.52.
  const Mat3 De = constitutive_matrix(E, nu, PlaneCondition::Strain);
  const Mat3 De_ref = {0.7 / 0.52, 0.3 / 0.52, 0.0, 0.3 / 0.52, 0.7 / 0.52,
                       0.0,        0.0,        0.0, 0.2 / 0.52};
  for (int i = 0; i < 9; ++i) {
    CHECK_NEAR(Ds[i], Ds_ref[i], 1e-15);
    CHECK_NEAR(De[i], De_ref[i], 1e-15);
  }
  CHECK(De[0] > Ds[0]);  // plane strain is stiffer
}

void test_stress_recovery() {
  std::mt19937 gen(777);
  const double a[3] = {0.3, 2.5e-3, -1.2e-3}, b[3] = {-0.1, 0.7e-3, -4.0e-3};
  const double exx = a[1], eyy = b[2], gxy = a[2] + b[1];
  const double E = 70e9, nu = 0.33;
  for (PlaneCondition plane : {PlaneCondition::Stress, PlaneCondition::Strain}) {
    const TriCoords xy = random_triangle(gen);
    const Vec3 e = cst_strain(xy, linear_field(xy, a, b));
    CHECK_NEAR(e[0], exx, 1e-12 * std::abs(exx));
    CHECK_NEAR(e[1], eyy, 1e-12 * std::abs(eyy));
    CHECK_NEAR(e[2], gxy, 1e-12 * std::abs(gxy));

    double sx, sy, txy;
    if (plane == PlaneCondition::Stress) {
      const double c = E / (1 - nu * nu);
      sx = c * (exx + nu * eyy);
      sy = c * (nu * exx + eyy);
      txy = c * (1 - nu) / 2 * gxy;
    } else {
      const double c = E / ((1 + nu) * (1 - 2 * nu));
      sx = c * ((1 - nu) * exx + nu * eyy);
      sy = c * (nu * exx + (1 - nu) * eyy);
      txy = c * (1 - 2 * nu) / 2 * gxy;
    }
    const Vec3 s = stress_from_strain(constitutive_matrix(E, nu, plane), e);
    CHECK_NEAR(s[0], sx, 1e-12 * std::abs(sx));
    CHECK_NEAR(s[1], sy, 1e-12 * std::abs(sy));
    CHECK_NEAR(s[2], txy, 1e-12 * std::abs(txy));
  }
}

void test_von_mises() {
  // Uniaxial (plane stress): vm = |sigma|; pure shear: vm = sqrt(3) tau.
  CHECK_NEAR(von_mises({100.0, 0.0, 0.0}, 0.3, PlaneCondition::Stress), 100.0, 1e-12);
  CHECK_NEAR(von_mises({0.0, 0.0, 50.0}, 0.3, PlaneCondition::Stress), 50.0 * std::sqrt(3.0),
             1e-12);
  // Plane strain, equibiaxial s: sz = 2 nu s, vm = |s - sz| = |s| (1 - 2 nu).
  const double s = 80.0, nu = 0.25;
  CHECK_NEAR(sigma_zz({s, s, 0.0}, nu, PlaneCondition::Strain), 2 * nu * s, 1e-12);
  CHECK_NEAR(sigma_zz({s, s, 0.0}, nu, PlaneCondition::Stress), 0.0, 0.0);
  CHECK_NEAR(von_mises({s, s, 0.0}, nu, PlaneCondition::Strain), s * (1 - 2 * nu), 1e-12);
}

}  // namespace

int main() {
  RUN_TEST(test_ke_reference_triangle);
  RUN_TEST(test_ke_properties);
  RUN_TEST(test_constitutive_closed_form);
  RUN_TEST(test_stress_recovery);
  RUN_TEST(test_von_mises);
  return test_summary();
}
