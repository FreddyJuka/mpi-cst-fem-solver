#pragma once

#include <array>

#include "cstfem/bc.hpp"
#include "cstfem/element.hpp"

namespace cstfem {

// Infinite plate with a circular hole of radius a under uniaxial sigma_inf along x
// (SPEC section 8). theta is measured from the x axis.
struct KirschParams {
  double sigma_inf = 0.0;
  double a = 0.0;
};

struct PolarStress {
  double rr = 0.0;
  double tt = 0.0;
  double rt = 0.0;
};

PolarStress kirsch_polar(const KirschParams& p, double r, double theta);

// (sigma_rr, sigma_tt, sigma_rt) at angle theta -> (sigma_xx, sigma_yy, tau_xy).
Vec3 polar_to_cartesian(const PolarStress& s, double theta);
// Inverse rotation.
PolarStress cartesian_to_polar(const Vec3& s, double theta);

// Cartesian analytical stress (sigma_xx, sigma_yy, tau_xy) at (x, y); requires r > 0.
Vec3 kirsch_stress(const KirschParams& p, double x, double y);

// Traction t = sigma . n of the analytical field, as a callback for add_traction().
TractionFn kirsch_traction(const KirschParams& p);

// Analytical displacement (u_x, u_y) at (x, y), r > 0, derived and verified symbolically by
// python/derive_kirsch_displacement.py (strains -> SPEC stresses, equilibrium, symmetry
// u_x = 0 on x = 0 and u_y = 0 on y = 0, so no rigid-body terms). With mu = E / (2 (1 + nu))
// and the Kolosov constant kappa = (3 - nu) / (1 + nu) (plane stress) or 3 - 4 nu (strain):
//   u_r = s / (8 mu) [ (kappa - 1) r + 2 a^2 / r
//                      + 2 (r + (kappa + 1) a^2 / r - a^4 / r^3) cos 2theta ]
//   u_t = -s / (4 mu) [ r + (kappa - 1) a^2 / r + a^4 / r^3 ] sin 2theta
std::array<double, 2> kirsch_displacement(const KirschParams& p, double E, double nu,
                                          PlaneCondition plane, double x, double y);

}  // namespace cstfem
