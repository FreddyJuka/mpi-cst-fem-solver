#pragma once

#include <optional>
#include <string>
#include <vector>

#include "cstfem/element.hpp"
#include "cstfem/kirsch.hpp"
#include "cstfem/mesh.hpp"
#include "cstfem/vtu.hpp"

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

// Kirsch error metrics. With exact Kirsch tractions on the outer edges there is no
// finite-width error, so all errors are pure discretization error.
struct KirschMetrics {
  // sqrt(sum_e A_e |s_h - s(x_c)|^2 / sum_e A_e |s(x_c)|^2), |s|^2 = sxx^2 + syy^2 + 2 txy^2.
  double l2_rel_stress_error = 0.0;
  // sqrt(int |u_h - u|^2 / int |u|^2): u_h linear on each triangle, 3-point (degree 2) rule.
  double l2_rel_displacement_error = 0.0;
  // sqrt(int (s_h - s)^T D^-1 (s_h - s) / int s^T D^-1 s), same quadrature.
  double energy_rel_error = 0.0;
  // Element whose centroid is closest to (0, a); no extrapolation.
  int scf_element = -1;
  double scf_x = 0.0, scf_y = 0.0;
  double scf_numerical = 0.0;           // sigma_xx,h / sigma_inf
  double scf_analytical_centroid = 0.0;  // sigma_xx(x_c) / sigma_inf
  double scf_theoretical = 3.0;
};

// Material data needed by the analytical displacement and the energy norm.
struct ElasticMaterial {
  double E = 0.0, nu = 0.0;
  PlaneCondition plane = PlaneCondition::Stress;
};

// `u_local`: nodal displacements of `mesh` (2 per node).
KirschMetrics kirsch_metrics(const Mesh& mesh, const std::vector<double>& u_local,
                             const ElementStresses& s, const KirschParams& p,
                             const ElasticMaterial& m);

// Building blocks of kirsch_metrics() for distributed use (sums are reduced over ranks).
struct KirschPartial {
  double err2 = 0.0;          // sum_e A_e |s_h - s(x_c)|^2
  double ref2 = 0.0;          // sum_e A_e |s(x_c)|^2
  double disp_err2 = 0.0;     // int |u_h - u|^2
  double disp_ref2 = 0.0;     // int |u|^2
  double energy_err2 = 0.0;   // int (s_h - s)^T D^-1 (s_h - s)
  double energy_ref2 = 0.0;   // int s^T D^-1 s
  int scf_element = -1;          // local element closest to (0, a)
  double scf_distance = 1e300;   // its centroid distance to (0, a)
};
KirschPartial kirsch_partial(const Mesh& mesh, const std::vector<double>& u_local,
                             const ElementStresses& s, const KirschParams& p,
                             const ElasticMaterial& m);
// Relative errors from (globally reduced) sums.
void set_relative_errors(KirschMetrics& out, const KirschPartial& sums);
// Fills the SCF fields of `m` from element e.
void set_scf(KirschMetrics& m, const ElementStresses& s, const KirschParams& p, int e);

// VTU arrays of one solution piece (serial: the whole mesh; distributed: one rank's local
// mesh, owned nodes first).
//   points: displacement (3 components, z = 0), owned (1/0: filter duplicates across pieces),
//           gmsh_tag (Int64)
//   cells:  sigma_xx, sigma_yy, tau_xy, von_mises, sigma_zz (plane strain only), partition_id,
//           element_id (original file-order index), and for Kirsch the analytical stresses at
//           the centroid and the errors (numerical - analytical).
struct SolutionPiece {
  std::vector<VtuArray> point_data, cell_data;
};
SolutionPiece solution_arrays(const Mesh& mesh, const std::vector<double>& u_local,
                              const ElementStresses& es, PlaneCondition plane, int num_owned,
                              int partition_id, const std::vector<int>& element_ids,
                              const std::optional<KirschParams>& kirsch);

// Kirsch CSV rows at element centroids, numerical next to analytical, flattened:
//   hole (elements adjacent to `hole` edges), 6 per row:
//     x, y, r, theta, sigma_tt_num, sigma_tt_exact
//   section x = 0 (elements adjacent to `left_sym` edges), 4 per row:
//     x, y, sigma_xx_num, sigma_xx_exact
constexpr int kHoleCsvCols = 6;
constexpr int kSectionCsvCols = 4;
std::vector<double> hole_csv_rows(const Mesh& mesh, const ElementStresses& s,
                                  const KirschParams& p);
std::vector<double> section_x0_csv_rows(const Mesh& mesh, const ElementStresses& s,
                                        const KirschParams& p);
// Write the rows sorted by theta (hole) / y (section), 17 significant digits.
void write_hole_csv(const std::string& path, std::vector<double> rows);
void write_section_x0_csv(const std::string& path, std::vector<double> rows);

}  // namespace cstfem
