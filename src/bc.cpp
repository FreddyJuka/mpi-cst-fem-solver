#include "cstfem/bc.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "cstfem/assembly.hpp"

namespace cstfem {

namespace {

std::string trim(const std::string& s) {
  const auto first = s.find_first_not_of(" \t");
  if (first == std::string::npos) return {};
  const auto last = s.find_last_not_of(" \t");
  return s.substr(first, last - first + 1);
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> parts;
  std::string item;
  std::istringstream ss(s);
  while (std::getline(ss, item, sep)) parts.push_back(trim(item));
  return parts;
}

double parse_number(const std::string& s, const std::string& context) {
  const char* begin = s.c_str();
  char* end = nullptr;
  const double v = std::strtod(begin, &end);
  if (s.empty() || end != begin + s.size() || !std::isfinite(v))
    throw std::runtime_error(context + ": '" + s + "' is not a number");
  return v;
}

// Parses "group:key=val,key=val ; group:key=val" into (group, [(key, value)]) entries.
using KeyValues = std::vector<std::pair<std::string, double>>;
std::vector<std::pair<std::string, KeyValues>> parse_group_list(const std::string& text,
                                                                const char* what) {
  std::vector<std::pair<std::string, KeyValues>> out;
  for (const auto& item : split(text, ';')) {
    if (item.empty()) continue;
    const auto colon = item.find(':');
    if (colon == std::string::npos)
      throw std::runtime_error(std::string(what) + ": expected 'group:key=value' in '" + item +
                               "'");
    const std::string group = trim(item.substr(0, colon));
    if (group.empty()) throw std::runtime_error(std::string(what) + ": empty group in '" + item + "'");
    KeyValues kv;
    for (const auto& assign : split(item.substr(colon + 1), ',')) {
      const auto eq = assign.find('=');
      if (eq == std::string::npos)
        throw std::runtime_error(std::string(what) + ": expected 'key=value' in '" + item + "'");
      const std::string key = trim(assign.substr(0, eq));
      kv.emplace_back(key, parse_number(trim(assign.substr(eq + 1)),
                                        std::string(what) + " '" + group + ":" + key + "'"));
    }
    if (kv.empty()) throw std::runtime_error(std::string(what) + ": no values for '" + group + "'");
    out.emplace_back(group, std::move(kv));
  }
  return out;
}

const std::vector<std::array<int, 2>>& group_edges(const Mesh& mesh, const std::string& group) {
  const auto it = mesh.boundary_edges.find(group);
  if (it == mesh.boundary_edges.end())
    throw std::runtime_error("unknown boundary group '" + group + "'");
  return it->second;
}

std::uint64_t edge_key(int a, int b) {
  if (a > b) std::swap(a, b);
  return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint32_t>(b);
}

}  // namespace

// ---- Dirichlet ----------------------------------------------------------------------------

std::vector<DirichletSpec> parse_dirichlet(const std::string& text) {
  std::vector<DirichletSpec> specs;
  for (const auto& [group, kv] : parse_group_list(text, "bc.dirichlet")) {
    DirichletSpec s;
    s.group = group;
    for (const auto& [key, value] : kv) {
      if (key != "ux" && key != "uy")
        throw std::runtime_error("bc.dirichlet: unknown component '" + key + "' for group '" +
                                 group + "' (use ux or uy)");
      bool& has = key == "ux" ? s.has_ux : s.has_uy;
      if (has)
        throw std::runtime_error("bc.dirichlet: '" + key + "' given twice for group '" + group +
                                 "'");
      has = true;
      (key == "ux" ? s.ux : s.uy) = value;
    }
    specs.push_back(s);
  }
  return specs;
}

void DirichletDofs::add(int dof, double value) {
  const auto [it, inserted] = values_.emplace(dof, value);
  if (!inserted && it->second != value) {
    std::ostringstream msg;
    msg.precision(17);
    msg << "conflicting Dirichlet values for node " << dof / 2 << " (0-based) component "
        << (dof % 2 == 0 ? "ux" : "uy") << ": " << it->second << " vs " << value;
    throw std::runtime_error(msg.str());
  }
}

void add_dirichlet_group(const Mesh& mesh, const std::string& group, int component,
                         const FieldFn& value, DirichletDofs& dofs) {
  for (const auto& edge : group_edges(mesh, group))
    for (int n : edge) dofs.add(dof_index(n, component), value(mesh.xy[2 * n], mesh.xy[2 * n + 1]));
}

void add_dirichlet_group(const Mesh& mesh, const std::string& group, int component, double value,
                         DirichletDofs& dofs) {
  add_dirichlet_group(mesh, group, component, [value](double, double) { return value; }, dofs);
}

void add_dirichlet(const Mesh& mesh, const std::vector<DirichletSpec>& specs,
                   DirichletDofs& dofs) {
  for (const auto& s : specs) {
    if (s.has_ux) add_dirichlet_group(mesh, s.group, 0, s.ux, dofs);
    if (s.has_uy) add_dirichlet_group(mesh, s.group, 1, s.uy, dofs);
  }
}

void apply_dirichlet(const DirichletDofs& dofs, CsrMatrix& K, std::vector<double>& b) {
  // Column-indexed flags and values, so the same code works for ghost columns later.
  std::vector<char> fixed(K.num_cols, 0);
  std::vector<double> ubar(K.num_cols, 0.0);
  for (const auto& [dof, value] : dofs.values()) {
    if (dof < 0 || dof >= K.num_cols)
      throw std::runtime_error("apply_dirichlet: DOF " + std::to_string(dof) + " out of range");
    fixed[dof] = 1;
    ubar[dof] = value;
  }
  for (int i = 0; i < K.num_rows; ++i) {
    const int ci = K.first_row + i;
    if (fixed[ci]) {
      const double diag = K.diagonal(i);
      if (!(diag > 0.0))
        throw std::runtime_error("apply_dirichlet: non-positive diagonal at DOF " +
                                 std::to_string(ci));
      for (int k = K.row_ptr[i]; k < K.row_ptr[i + 1]; ++k)
        if (K.cols[k] != ci) K.vals[k] = 0.0;
      b[i] = diag * ubar[ci];
    } else {
      for (int k = K.row_ptr[i]; k < K.row_ptr[i + 1]; ++k) {
        if (fixed[K.cols[k]]) {
          b[i] -= K.vals[k] * ubar[K.cols[k]];
          K.vals[k] = 0.0;
        }
      }
    }
  }
}

// ---- Neumann ------------------------------------------------------------------------------

std::vector<TractionSpec> parse_traction(const std::string& text) {
  std::vector<TractionSpec> specs;
  for (const auto& [group, kv] : parse_group_list(text, "bc.traction")) {
    TractionSpec s;
    s.group = group;
    std::set<std::string> seen;
    for (const auto& [key, value] : kv) {
      if (key != "tx" && key != "ty")
        throw std::runtime_error("bc.traction: unknown component '" + key + "' for group '" +
                                 group + "' (use tx or ty)");
      if (!seen.insert(key).second)
        throw std::runtime_error("bc.traction: '" + key + "' given twice for group '" + group +
                                 "'");
      (key == "tx" ? s.tx : s.ty) = value;
    }
    specs.push_back(s);
  }
  return specs;
}

void add_traction(const Mesh& mesh, const std::string& group, double thickness,
                  const TractionFn& traction, std::vector<double>& f) {
  const auto& edges = group_edges(mesh, group);

  // Opposite node of each group edge, from the triangles adjacent to it.
  std::unordered_map<std::uint64_t, int> opposite;
  std::unordered_map<std::uint64_t, int> count;
  opposite.reserve(edges.size());
  for (const auto& e : edges) {
    opposite.emplace(edge_key(e[0], e[1]), -1);
    count.emplace(edge_key(e[0], e[1]), 0);
  }
  for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
    const int* n = &mesh.triangles[3 * t];
    for (int k = 0; k < 3; ++k) {
      const auto it = opposite.find(edge_key(n[k], n[(k + 1) % 3]));
      if (it != opposite.end()) {
        it->second = n[(k + 2) % 3];
        ++count[it->first];
      }
    }
  }

  // 2-point Gauss rule on [-1, 1].
  const double g = 1.0 / std::sqrt(3.0);
  const double xi[2] = {-g, g};

  for (const auto& e : edges) {
    const int a = e[0], b = e[1];
    const auto key = edge_key(a, b);
    if (count[key] != 1)
      throw std::runtime_error("traction group '" + group + "': edge (" +
                               std::to_string(mesh.node_tags[a]) + ", " +
                               std::to_string(mesh.node_tags[b]) + ") is adjacent to " +
                               std::to_string(count[key]) + " triangles (expected 1)");
    const double xa = mesh.xy[2 * a], ya = mesh.xy[2 * a + 1];
    const double xb = mesh.xy[2 * b], yb = mesh.xy[2 * b + 1];
    const double len = std::hypot(xb - xa, yb - ya);
    double nx = (yb - ya) / len, ny = -(xb - xa) / len;
    // Outward: pointing away from the opposite node of the adjacent triangle.
    const int c = opposite[key];
    if (nx * (mesh.xy[2 * c] - xa) + ny * (mesh.xy[2 * c + 1] - ya) > 0.0) {
      nx = -nx;
      ny = -ny;
    }
    for (double q : xi) {
      const double Na = 0.5 * (1.0 - q), Nb = 0.5 * (1.0 + q);
      const double x = Na * xa + Nb * xb, y = Na * ya + Nb * yb;
      const auto t = traction(x, y, nx, ny);
      const double w = thickness * 0.5 * len;  // Gauss weight 1 times the Jacobian len/2
      f[dof_index(a, 0)] += w * Na * t[0];
      f[dof_index(a, 1)] += w * Na * t[1];
      f[dof_index(b, 0)] += w * Nb * t[0];
      f[dof_index(b, 1)] += w * Nb * t[1];
    }
  }
}

void add_traction(const Mesh& mesh, const std::string& group, double thickness, double tx,
                  double ty, std::vector<double>& f) {
  add_traction(
      mesh, group, thickness,
      [tx, ty](double, double, double, double) { return std::array<double, 2>{tx, ty}; }, f);
}

}  // namespace cstfem
