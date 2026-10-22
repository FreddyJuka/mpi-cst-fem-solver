#pragma once

#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "cstfem/case.hpp"
#include "cstfem/element.hpp"
#include "cstfem/kirsch.hpp"
#include "cstfem/mesh.hpp"
#include "cstfem/solver.hpp"

namespace cstfem {

// Validated solver settings of a case file.
struct CaseSettings {
  std::string mesh_path, output_dir, plane_name;
  PlaneCondition plane = PlaneCondition::Stress;
  double E = 0, nu = 0, thickness = 0;
  SolverOptions solver;
};
CaseSettings read_case_settings(const CaseFile& c);

// Keys understood by the solver; others trigger a warning.
const std::vector<std::string>& known_case_keys();

// Adds the loads of `bc.traction` to f:
//   kirsch_exact            -> Kirsch traction on `right` and `top` (hole stays traction-free),
//                              using kirsch.sigma_inf and kirsch.a; returns those parameters.
//   group:tx=..,ty=.. ; ... -> constant tractions; returns nullopt.
std::optional<KirschParams> add_case_tractions(const CaseFile& c, const Mesh& mesh,
                                               double thickness, std::vector<double>& f);

// Serial end-to-end run: case -> mesh -> assembly -> BCs -> solve -> stresses ->
// VTU/CSV/timings.json in output_dir. Returns 0 on success, 2 if the solver did not
// converge, 1 on any other error (message on `err`).
int run_serial_case(const std::string& case_path, std::ostream& out, std::ostream& err);

}  // namespace cstfem
