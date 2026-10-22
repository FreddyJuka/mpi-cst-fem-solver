#pragma once

#include <string>
#include <vector>

#include "cstfem/element.hpp"
#include "cstfem/kirsch.hpp"
#include "cstfem/mesh.hpp"

namespace cstfem {

// Constant stress of every element plus its centroid and area.
struct ElementStresses {
  std::vector<double> sxx, syy, txy, von_mises, szz;
  std::vector<double> cx, cy, area;
};

ElementStresses compute_element_stresses(const Mesh& mesh, const std::vector<double>& u,
                                         const Mat3& D, double nu, PlaneCondition plane);

// Triangles that contain an edge of line group `group`, sorted by index.
std::vector<int> elements_adjacent_to_group(const Mesh& mesh, const std::string& group);

struct KirschMetrics {
  // sqrt(sum_e A_e |s_h - s(x_c)|^2 / sum_e A_e |s(x_c)|^2), |s|^2 = sxx^2 + syy^2 + 2 txy^2.
  double l2_rel_stress_error = 0.0;
  // Element whose centroid is closest to (0, a); no extrapolation.
  int scf_element = -1;
  double scf_x = 0.0, scf_y = 0.0;
  double scf_numerical = 0.0;           // sigma_xx,h / sigma_inf
  double scf_analytical_centroid = 0.0;  // sigma_xx(x_c) / sigma_inf
  double scf_theoretical = 3.0;
};

KirschMetrics kirsch_metrics(const ElementStresses& s, const KirschParams& p);

// Building blocks of kirsch_metrics() for distributed use (sums are reduced over ranks).
struct KirschPartial {
  double err2 = 0.0;  // sum_e A_e |s_h - s(x_c)|^2
  double ref2 = 0.0;  // sum_e A_e |s(x_c)|^2
  int scf_element = -1;          // local element closest to (0, a)
  double scf_distance = 1e300;   // its centroid distance to (0, a)
};
KirschPartial kirsch_partial(const ElementStresses& s, const KirschParams& p);
// Fills the SCF fields of `m` from element e.
void set_scf(KirschMetrics& m, const ElementStresses& s, const KirschParams& p, int e);

// Centroid values next to the analytical field (17 significant digits).
// hole_sigma_tt.csv: x, y, r, theta, sigma_tt_num, sigma_tt_exact (sorted by theta).
void write_hole_csv(const std::string& path, const Mesh& mesh, const ElementStresses& s,
                    const KirschParams& p);
// section_x0_sigma_xx.csv: x, y, sigma_xx_num, sigma_xx_exact (sorted by y).
void write_section_x0_csv(const std::string& path, const Mesh& mesh, const ElementStresses& s,
                          const KirschParams& p);

}  // namespace cstfem
