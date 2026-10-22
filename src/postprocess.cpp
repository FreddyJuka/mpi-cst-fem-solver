#include "cstfem/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>

#include "cstfem/assembly.hpp"

namespace cstfem {

namespace {

std::uint64_t edge_key(int a, int b) {
  if (a > b) std::swap(a, b);
  return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint32_t>(b);
}

double norm2(double sxx, double syy, double txy) { return sxx * sxx + syy * syy + 2 * txy * txy; }

struct File {
  explicit File(const std::string& path) : f(std::fopen(path.c_str(), "w")) {
    if (!f) throw std::runtime_error(path + ": cannot open for writing");
  }
  ~File() { std::fclose(f); }
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  std::FILE* f;
};

}  // namespace

ElementStresses compute_element_stresses(const Mesh& mesh, const std::vector<double>& u,
                                         const Mat3& D, double nu, PlaneCondition plane) {
  const std::size_t ne = mesh.num_triangles();
  ElementStresses s;
  for (auto* v : {&s.sxx, &s.syy, &s.txy, &s.von_mises, &s.szz, &s.cx, &s.cy, &s.area})
    v->resize(ne);
  for (std::size_t e = 0; e < ne; ++e) {
    const TriCoords xy = element_coords(mesh, e);
    Vec6 ue;
    for (int k = 0; k < 3; ++k) {
      const int n = mesh.triangles[3 * e + k];
      ue[2 * k] = u[dof_index(n, 0)];
      ue[2 * k + 1] = u[dof_index(n, 1)];
    }
    const Vec3 sig = stress_from_strain(D, cst_strain(xy, ue));
    s.sxx[e] = sig[0];
    s.syy[e] = sig[1];
    s.txy[e] = sig[2];
    s.szz[e] = sigma_zz(sig, nu, plane);
    s.von_mises[e] = von_mises(sig, nu, plane);
    s.cx[e] = (xy[0] + xy[2] + xy[4]) / 3.0;
    s.cy[e] = (xy[1] + xy[3] + xy[5]) / 3.0;
    s.area[e] = std::abs(cst_signed_area(xy));
  }
  return s;
}

std::vector<int> elements_adjacent_to_group(const Mesh& mesh, const std::string& group) {
  const auto it = mesh.boundary_edges.find(group);
  if (it == mesh.boundary_edges.end())
    throw std::runtime_error("unknown boundary group '" + group + "'");
  std::unordered_set<std::uint64_t> edges;
  for (const auto& e : it->second) edges.insert(edge_key(e[0], e[1]));
  std::vector<int> out;
  for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
    const int* n = &mesh.triangles[3 * t];
    for (int k = 0; k < 3; ++k) {
      if (edges.count(edge_key(n[k], n[(k + 1) % 3]))) {
        out.push_back(static_cast<int>(t));
        break;
      }
    }
  }
  return out;
}

KirschPartial kirsch_partial(const ElementStresses& s, const KirschParams& p) {
  KirschPartial k;
  for (std::size_t e = 0; e < s.sxx.size(); ++e) {
    const Vec3 ex = kirsch_stress(p, s.cx[e], s.cy[e]);
    k.err2 += s.area[e] * norm2(s.sxx[e] - ex[0], s.syy[e] - ex[1], s.txy[e] - ex[2]);
    k.ref2 += s.area[e] * norm2(ex[0], ex[1], ex[2]);
    const double d = std::hypot(s.cx[e], s.cy[e] - p.a);
    if (d < k.scf_distance) {
      k.scf_distance = d;
      k.scf_element = static_cast<int>(e);
    }
  }
  return k;
}

void set_scf(KirschMetrics& m, const ElementStresses& s, const KirschParams& p, int e) {
  m.scf_element = e;
  m.scf_x = s.cx[e];
  m.scf_y = s.cy[e];
  m.scf_numerical = s.sxx[e] / p.sigma_inf;
  m.scf_analytical_centroid = kirsch_stress(p, s.cx[e], s.cy[e])[0] / p.sigma_inf;
}

KirschMetrics kirsch_metrics(const ElementStresses& s, const KirschParams& p) {
  const KirschPartial k = kirsch_partial(s, p);
  KirschMetrics m;
  m.l2_rel_stress_error = std::sqrt(k.err2 / k.ref2);
  if (k.scf_element >= 0) set_scf(m, s, p, k.scf_element);
  return m;
}

SolutionPiece solution_arrays(const Mesh& mesh, const std::vector<double>& u_local,
                              const ElementStresses& es, PlaneCondition plane, int num_owned,
                              int partition_id, const std::vector<int>& element_ids,
                              const std::optional<KirschParams>& kirsch) {
  const std::size_t nn = mesh.num_nodes(), ne = mesh.num_triangles();
  SolutionPiece out;
  VtuArray disp{"displacement", 3, std::vector<double>(3 * nn, 0.0), {}, {}};
  VtuArray owned{"owned", 1, {}, std::vector<std::int32_t>(nn, 0), {}};
  VtuArray tags{"gmsh_tag", 1, {}, {}, std::vector<std::int64_t>(nn)};
  for (std::size_t n = 0; n < nn; ++n) {
    disp.f64[3 * n] = u_local[2 * n];
    disp.f64[3 * n + 1] = u_local[2 * n + 1];
    owned.i32[n] = static_cast<int>(n) < num_owned ? 1 : 0;
    tags.i64[n] = static_cast<std::int64_t>(mesh.node_tags[n]);
  }
  out.point_data = {std::move(disp), std::move(owned), std::move(tags)};

  auto& cd = out.cell_data;
  cd = {{"sigma_xx", 1, es.sxx, {}, {}},
        {"sigma_yy", 1, es.syy, {}, {}},
        {"tau_xy", 1, es.txy, {}, {}},
        {"von_mises", 1, es.von_mises, {}, {}}};
  if (plane == PlaneCondition::Strain) cd.push_back({"sigma_zz", 1, es.szz, {}, {}});
  cd.push_back({"partition_id", 1, {}, std::vector<std::int32_t>(ne, partition_id), {}});
  cd.push_back({"element_id", 1, {}, {element_ids.begin(), element_ids.end()}, {}});
  if (kirsch) {
    VtuArray ex{"sigma_xx_exact", 1, std::vector<double>(ne), {}, {}};
    VtuArray ey{"sigma_yy_exact", 1, std::vector<double>(ne), {}, {}};
    VtuArray et{"tau_xy_exact", 1, std::vector<double>(ne), {}, {}};
    VtuArray dx{"sigma_xx_error", 1, std::vector<double>(ne), {}, {}};
    VtuArray dy{"sigma_yy_error", 1, std::vector<double>(ne), {}, {}};
    VtuArray dt{"tau_xy_error", 1, std::vector<double>(ne), {}, {}};
    for (std::size_t e = 0; e < ne; ++e) {
      const Vec3 a = kirsch_stress(*kirsch, es.cx[e], es.cy[e]);
      ex.f64[e] = a[0];
      ey.f64[e] = a[1];
      et.f64[e] = a[2];
      dx.f64[e] = es.sxx[e] - a[0];
      dy.f64[e] = es.syy[e] - a[1];
      dt.f64[e] = es.txy[e] - a[2];
    }
    for (auto* a : {&ex, &ey, &et, &dx, &dy, &dt}) cd.push_back(std::move(*a));
  }
  return out;
}

std::vector<double> hole_csv_rows(const Mesh& mesh, const ElementStresses& s,
                                  const KirschParams& p) {
  std::vector<double> rows;
  for (int e : elements_adjacent_to_group(mesh, "hole")) {
    const double x = s.cx[e], y = s.cy[e], r = std::hypot(x, y), th = std::atan2(y, x);
    rows.insert(rows.end(), {x, y, r, th,
                             cartesian_to_polar({s.sxx[e], s.syy[e], s.txy[e]}, th).tt,
                             kirsch_polar(p, r, th).tt});
  }
  return rows;
}

std::vector<double> section_x0_csv_rows(const Mesh& mesh, const ElementStresses& s,
                                        const KirschParams& p) {
  std::vector<double> rows;
  for (int e : elements_adjacent_to_group(mesh, "left_sym"))
    rows.insert(rows.end(),
                {s.cx[e], s.cy[e], s.sxx[e], kirsch_stress(p, s.cx[e], s.cy[e])[0]});
  return rows;
}

namespace {

void write_csv(const std::string& path, const char* header, std::vector<double> rows, int cols,
               int sort_col) {
  const std::size_t n = rows.size() / cols;
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return rows[a * cols + sort_col] < rows[b * cols + sort_col];
  });
  File out(path);
  std::fprintf(out.f, "%s\n", header);
  for (std::size_t i : order) {
    for (int c = 0; c < cols; ++c)
      std::fprintf(out.f, c + 1 < cols ? "%.17g," : "%.17g\n", rows[i * cols + c]);
  }
}

}  // namespace

void write_hole_csv(const std::string& path, std::vector<double> rows) {
  write_csv(path, "x,y,r,theta,sigma_tt_num,sigma_tt_exact", std::move(rows), kHoleCsvCols, 3);
}

void write_section_x0_csv(const std::string& path, std::vector<double> rows) {
  write_csv(path, "x,y,sigma_xx_num,sigma_xx_exact", std::move(rows), kSectionCsvCols, 1);
}

}  // namespace cstfem
