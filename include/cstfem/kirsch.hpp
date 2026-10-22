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

}  // namespace cstfem
