#pragma once

#include <array>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "cstfem/csr.hpp"
#include "cstfem/mesh.hpp"

namespace cstfem {

// ---- Dirichlet ----------------------------------------------------------------------------

// One entry of `bc.dirichlet = group:ux=val,uy=val ; group:uy=val`.
struct DirichletSpec {
  std::string group;
  bool has_ux = false;
  bool has_uy = false;
  double ux = 0.0;
  double uy = 0.0;
};

std::vector<DirichletSpec> parse_dirichlet(const std::string& text);

// Prescribed DOF values keyed by DOF (column) index.
class DirichletDofs {
public:
  // Adds dof = value. The same DOF may be added again with the identical value (e.g. a corner
  // node shared by two groups); a different value throws.
  void add(int dof, double value);
  const std::map<int, double>& values() const { return values_; }
  std::size_t size() const { return values_.size(); }

private:
  std::map<int, double> values_;
};

using FieldFn = std::function<double(double x, double y)>;

// Prescribes component (0 = ux, 1 = uy) on every node of line group `group`.
void add_dirichlet_group(const Mesh& mesh, const std::string& group, int component, double value,
                         DirichletDofs& dofs);
void add_dirichlet_group(const Mesh& mesh, const std::string& group, int component,
                         const FieldFn& value, DirichletDofs& dofs);
void add_dirichlet(const Mesh& mesh, const std::vector<DirichletSpec>& specs,
                   DirichletDofs& dofs);

// Symmetric elimination: for each constrained DOF j with value u_j,
// b -= K(:, j) u_j; row j and column j are zeroed except the diagonal, which keeps its value
// K_jj, and b_j = K_jj u_j. K stays symmetric positive definite.
void apply_dirichlet(const DirichletDofs& dofs, CsrMatrix& K, std::vector<double>& b);

// ---- Neumann ------------------------------------------------------------------------------

// One entry of `bc.traction = group:tx=val,ty=val ; ...`.
struct TractionSpec {
  std::string group;
  double tx = 0.0;
  double ty = 0.0;
};

std::vector<TractionSpec> parse_traction(const std::string& text);

// Traction (tx, ty) at point (x, y) with outward unit normal (nx, ny).
using TractionFn = std::function<std::array<double, 2>(double x, double y, double nx, double ny)>;

// Adds consistent nodal forces of a traction on line group `group` to f (2 per node).
// 2-point Gauss quadrature per edge, times thickness. The outward normal of each edge is
// taken from its adjacent triangle, independent of the edge orientation in the file.
void add_traction(const Mesh& mesh, const std::string& group, double thickness,
                  const TractionFn& traction, std::vector<double>& f);
void add_traction(const Mesh& mesh, const std::string& group, double thickness, double tx,
                  double ty, std::vector<double>& f);

}  // namespace cstfem
