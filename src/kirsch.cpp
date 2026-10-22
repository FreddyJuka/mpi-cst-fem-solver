#include "cstfem/kirsch.hpp"

#include <cmath>

namespace cstfem {

PolarStress kirsch_polar(const KirschParams& p, double r, double theta) {
  const double s = p.sigma_inf;
  const double q2 = (p.a * p.a) / (r * r);  // a^2 / r^2
  const double q4 = q2 * q2;                 // a^4 / r^4
  const double c2 = std::cos(2.0 * theta), s2 = std::sin(2.0 * theta);
  PolarStress out;
  out.rr = 0.5 * s * (1.0 - q2) + 0.5 * s * (1.0 - 4.0 * q2 + 3.0 * q4) * c2;
  out.tt = 0.5 * s * (1.0 + q2) - 0.5 * s * (1.0 + 3.0 * q4) * c2;
  out.rt = -0.5 * s * (1.0 + 2.0 * q2 - 3.0 * q4) * s2;
  return out;
}

Vec3 polar_to_cartesian(const PolarStress& s, double theta) {
  const double c = std::cos(theta), n = std::sin(theta);
  return {s.rr * c * c + s.tt * n * n - 2.0 * s.rt * n * c,
          s.rr * n * n + s.tt * c * c + 2.0 * s.rt * n * c,
          (s.rr - s.tt) * n * c + s.rt * (c * c - n * n)};
}

PolarStress cartesian_to_polar(const Vec3& s, double theta) {
  const double c = std::cos(theta), n = std::sin(theta);
  PolarStress out;
  out.rr = s[0] * c * c + s[1] * n * n + 2.0 * s[2] * n * c;
  out.tt = s[0] * n * n + s[1] * c * c - 2.0 * s[2] * n * c;
  out.rt = (s[1] - s[0]) * n * c + s[2] * (c * c - n * n);
  return out;
}

Vec3 kirsch_stress(const KirschParams& p, double x, double y) {
  const double r = std::hypot(x, y), theta = std::atan2(y, x);
  return polar_to_cartesian(kirsch_polar(p, r, theta), theta);
}

TractionFn kirsch_traction(const KirschParams& p) {
  return [p](double x, double y, double nx, double ny) {
    const Vec3 s = kirsch_stress(p, x, y);
    return std::array<double, 2>{s[0] * nx + s[2] * ny, s[2] * nx + s[1] * ny};
  };
}

}  // namespace cstfem
