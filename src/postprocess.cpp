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

void write_hole_csv(const std::string& path, const Mesh& mesh, const ElementStresses& s,
                    const KirschParams& p) {
  auto elems = elements_adjacent_to_group(mesh, "hole");
  auto theta = [&](int e) { return std::atan2(s.cy[e], s.cx[e]); };
  std::sort(elems.begin(), elems.end(), [&](int a, int b) { return theta(a) < theta(b); });
  File out(path);
  std::fprintf(out.f, "x,y,r,theta,sigma_tt_num,sigma_tt_exact\n");
  for (int e : elems) {
    const double x = s.cx[e], y = s.cy[e], r = std::hypot(x, y), th = theta(e);
    const double num = cartesian_to_polar({s.sxx[e], s.syy[e], s.txy[e]}, th).tt;
    const double exact = kirsch_polar(p, r, th).tt;
    std::fprintf(out.f, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", x, y, r, th, num, exact);
  }
}

void write_section_x0_csv(const std::string& path, const Mesh& mesh, const ElementStresses& s,
                          const KirschParams& p) {
  auto elems = elements_adjacent_to_group(mesh, "left_sym");
  std::sort(elems.begin(), elems.end(), [&](int a, int b) { return s.cy[a] < s.cy[b]; });
  File out(path);
  std::fprintf(out.f, "x,y,sigma_xx_num,sigma_xx_exact\n");
  for (int e : elems)
    std::fprintf(out.f, "%.17g,%.17g,%.17g,%.17g\n", s.cx[e], s.cy[e], s.sxx[e],
                 kirsch_stress(p, s.cx[e], s.cy[e])[0]);
}

}  // namespace cstfem
