#pragma once

#include <array>

namespace cstfem {

enum class PlaneCondition { Stress, Strain };

// Row-major fixed-size matrices and vectors.
using Mat3 = std::array<double, 9>;     // 3x3
using Mat36 = std::array<double, 18>;   // 3x6
using Mat6 = std::array<double, 36>;    // 6x6
using Vec3 = std::array<double, 3>;     // (xx, yy, xy) strain or stress
using Vec6 = std::array<double, 6>;     // element DOFs (u1, v1, u2, v2, u3, v3)
using TriCoords = std::array<double, 6>;  // (x1, y1, x2, y2, x3, y3)

// Isotropic constitutive matrix D (SPEC section 2).
Mat3 constitutive_matrix(double E, double nu, PlaneCondition plane);

// Signed area: positive for counter-clockwise node order.
double cst_signed_area(const TriCoords& xy);

// Strain-displacement matrix B (3x6). Strain is (exx, eyy, gamma_xy), engineering shear.
Mat36 cst_B(const TriCoords& xy);

// Element stiffness Ke = t * |A| * B^T D B.
Mat6 cst_stiffness(const TriCoords& xy, const Mat3& D, double thickness);

// Constant element strain B * ue.
Vec3 cst_strain(const TriCoords& xy, const Vec6& ue);

// Stress D * strain: (sxx, syy, txy).
Vec3 stress_from_strain(const Mat3& D, const Vec3& strain);

// Out-of-plane stress: nu * (sxx + syy) in plane strain, 0 in plane stress.
double sigma_zz(const Vec3& stress, double nu, PlaneCondition plane);

// Von Mises stress; includes sigma_zz in plane strain.
double von_mises(const Vec3& stress, double nu, PlaneCondition plane);

}  // namespace cstfem
