#include "cstfem/element.hpp"

#include <cmath>

namespace cstfem {

Mat3 constitutive_matrix(double E, double nu, PlaneCondition plane) {
  if (plane == PlaneCondition::Stress) {
    const double c = E / (1.0 - nu * nu);
    return {c, c * nu, 0.0, c * nu, c, 0.0, 0.0, 0.0, c * (1.0 - nu) / 2.0};
  }
  const double c = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
  return {c * (1.0 - nu), c * nu, 0.0, c * nu, c * (1.0 - nu), 0.0,
          0.0,            0.0,    c * (1.0 - 2.0 * nu) / 2.0};
}

double cst_signed_area(const TriCoords& xy) {
  return 0.5 * ((xy[2] - xy[0]) * (xy[5] - xy[1]) - (xy[4] - xy[0]) * (xy[3] - xy[1]));
}

Mat36 cst_B(const TriCoords& xy) {
  // b_i = y_j - y_k, c_i = x_k - x_j for cyclic (i, j, k); B = 1/(2A) [b_i 0; 0 c_i; c_i b_i].
  const double x1 = xy[0], y1 = xy[1], x2 = xy[2], y2 = xy[3], x3 = xy[4], y3 = xy[5];
  const double b[3] = {y2 - y3, y3 - y1, y1 - y2};
  const double c[3] = {x3 - x2, x1 - x3, x2 - x1};
  const double inv2A = 1.0 / (2.0 * cst_signed_area(xy));
  Mat36 B{};
  for (int i = 0; i < 3; ++i) {
    B[0 * 6 + 2 * i] = b[i] * inv2A;
    B[1 * 6 + 2 * i + 1] = c[i] * inv2A;
    B[2 * 6 + 2 * i] = c[i] * inv2A;
    B[2 * 6 + 2 * i + 1] = b[i] * inv2A;
  }
  return B;
}

Mat6 cst_stiffness(const TriCoords& xy, const Mat3& D, double thickness) {
  const Mat36 B = cst_B(xy);
  Mat36 DB{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < 3; ++k) DB[i * 6 + j] += D[i * 3 + k] * B[k * 6 + j];

  const double scale = thickness * std::abs(cst_signed_area(xy));
  Mat6 K{};
  // Compute the upper triangle and mirror it so Ke is exactly symmetric.
  for (int i = 0; i < 6; ++i) {
    for (int j = i; j < 6; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += B[k * 6 + i] * DB[k * 6 + j];
      K[i * 6 + j] = K[j * 6 + i] = scale * s;
    }
  }
  return K;
}

Vec3 cst_strain(const TriCoords& xy, const Vec6& ue) {
  const Mat36 B = cst_B(xy);
  Vec3 e{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 6; ++j) e[i] += B[i * 6 + j] * ue[j];
  return e;
}

Vec3 stress_from_strain(const Mat3& D, const Vec3& strain) {
  Vec3 s{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) s[i] += D[i * 3 + j] * strain[j];
  return s;
}

double sigma_zz(const Vec3& stress, double nu, PlaneCondition plane) {
  return plane == PlaneCondition::Strain ? nu * (stress[0] + stress[1]) : 0.0;
}

double von_mises(const Vec3& stress, double nu, PlaneCondition plane) {
  const double sx = stress[0], sy = stress[1], txy = stress[2];
  const double sz = sigma_zz(stress, nu, plane);
  const double j2x2 = (sx - sy) * (sx - sy) + (sy - sz) * (sy - sz) + (sz - sx) * (sz - sx);
  return std::sqrt(0.5 * j2x2 + 3.0 * txy * txy);
}

}  // namespace cstfem
