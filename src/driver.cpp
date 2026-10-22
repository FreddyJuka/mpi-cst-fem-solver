#include "cstfem/driver.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "cstfem/assembly.hpp"
#include "cstfem/bc.hpp"
#include "cstfem/config.hpp"
#include "cstfem/postprocess.hpp"
#include "cstfem/solver.hpp"
#include "cstfem/vtu.hpp"

namespace cstfem {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// JSON number with full precision; non-finite values become null.
std::string num(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

std::string str(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

struct Settings {
  std::string mesh_path, output_dir, plane_name;
  PlaneCondition plane = PlaneCondition::Stress;
  double E = 0, nu = 0, thickness = 0;
  SolverOptions solver;
};

Settings read_settings(const CaseFile& c) {
  Settings s;
  s.mesh_path = c.get_string("mesh");
  s.output_dir = c.get_string("output_dir", "results");
  s.plane_name = c.get_string("plane");
  if (s.plane_name == "plane_stress")
    s.plane = PlaneCondition::Stress;
  else if (s.plane_name == "plane_strain")
    s.plane = PlaneCondition::Strain;
  else
    throw std::runtime_error(c.source() + ": plane must be plane_stress or plane_strain, got '" +
                             s.plane_name + "'");
  s.E = c.get_double("E");
  s.nu = c.get_double("nu");
  s.thickness = c.get_double("thickness");
  if (!(s.E > 0)) throw std::runtime_error(c.source() + ": E must be positive");
  if (!(s.nu > -1.0 && s.nu < 0.5)) throw std::runtime_error(c.source() + ": nu must be in (-1, 0.5)");
  if (!(s.thickness > 0)) throw std::runtime_error(c.source() + ": thickness must be positive");

  const std::string partitioner = c.get_string("partitioner", "metis");
  if (partitioner == "parmetis" && !USE_PARMETIS)
    throw std::runtime_error(c.source() +
                             ": partitioner = parmetis, but ParMETIS was not compiled in "
                             "(configure with -DUSE_PARMETIS=ON)");
  if (partitioner != "metis" && partitioner != "parmetis")
    throw std::runtime_error(c.source() + ": partitioner must be metis or parmetis");
  const std::string solver = c.get_string("solver", "cg_jacobi");
  if (solver != "cg_jacobi")
    throw std::runtime_error(c.source() + ": solver '" + solver +
                             "' is not available (only cg_jacobi in this version)");
  s.solver.rtol = c.get_double("rtol", 1e-10);
  s.solver.max_iter = c.get_int("max_iter", 20000);
  return s;
}

}  // namespace

const std::vector<std::string>& known_case_keys() {
  static const std::vector<std::string> keys = {
      "mesh",   "plane",    "E",          "nu",           "thickness",   "partitioner",
      "solver", "rtol",     "max_iter",   "output_dir",   "bc.dirichlet", "bc.traction",
      "kirsch.sigma_inf", "kirsch.a"};
  return keys;
}

std::optional<KirschParams> add_case_tractions(const CaseFile& c, const Mesh& mesh,
                                               double thickness, std::vector<double>& f) {
  const std::string spec = c.get_string("bc.traction", "");
  if (spec == "kirsch_exact") {
    const KirschParams p{c.get_double("kirsch.sigma_inf"), c.get_double("kirsch.a")};
    if (!(p.a > 0)) throw std::runtime_error(c.source() + ": kirsch.a must be positive");
    const TractionFn t = kirsch_traction(p);
    add_traction(mesh, "right", thickness, t, f);
    add_traction(mesh, "top", thickness, t, f);
    return p;
  }
  for (const auto& tr : parse_traction(spec)) add_traction(mesh, tr.group, thickness, tr.tx, tr.ty, f);
  return std::nullopt;
}

int run_serial_case(const std::string& case_path, std::ostream& out, std::ostream& err) {
  try {
    const auto t_total = Clock::now();

    // ---- read
    auto t0 = Clock::now();
    const CaseFile c = CaseFile::load(case_path);
    c.warn_unknown_keys(known_case_keys(), err);
    const Settings s = read_settings(c);
    const Mesh mesh = read_gmsh(s.mesh_path);
    const double t_read = seconds_since(t0);
    out << "mesh " << s.mesh_path << ": " << mesh.num_nodes() << " nodes, "
        << mesh.num_triangles() << " triangles, " << 2 * mesh.num_nodes() << " DOFs\n";

    // ---- assembly
    t0 = Clock::now();
    const Mat3 D = constitutive_matrix(s.E, s.nu, s.plane);
    CsrMatrix K = build_stiffness_pattern(mesh);
    assemble_stiffness(mesh, D, s.thickness, K);
    const double t_assembly = seconds_since(t0);

    // ---- boundary conditions
    t0 = Clock::now();
    std::vector<double> f(2 * mesh.num_nodes(), 0.0);
    const auto kirsch = add_case_tractions(c, mesh, s.thickness, f);
    DirichletDofs dofs;
    add_dirichlet(mesh, parse_dirichlet(c.get_string("bc.dirichlet", "")), dofs);
    apply_dirichlet(dofs, K, f);
    const double t_bcs = seconds_since(t0);

    // ---- solve
    t0 = Clock::now();
    CgJacobiSolver solver(s.solver);
    solver.setup(K);
    std::vector<double> u;
    const SolverStats st = solver.solve(f, u);
    const double t_solve = seconds_since(t0);
    out << "cg_jacobi: " << st.message << "; true relative residual " << st.true_rel_residual
        << "\n";
    if (!st.warning.empty()) err << st.warning << "\n";
    if (!st.converged) err << "error: solver did not converge: " << st.message << "\n";

    // ---- stresses and output
    t0 = Clock::now();
    std::filesystem::create_directories(s.output_dir);
    const std::filesystem::path dir(s.output_dir);
    const ElementStresses es = compute_element_stresses(mesh, u, D, s.nu, s.plane);
    const std::size_t ne = mesh.num_triangles();

    std::vector<VtuArray> point_data(1);
    point_data[0].name = "displacement";
    point_data[0].components = 3;
    point_data[0].f64.resize(3 * mesh.num_nodes(), 0.0);
    for (std::size_t n = 0; n < mesh.num_nodes(); ++n) {
      point_data[0].f64[3 * n] = u[2 * n];
      point_data[0].f64[3 * n + 1] = u[2 * n + 1];
    }
    std::vector<VtuArray> cell_data = {{"sigma_xx", 1, es.sxx, {}},
                                       {"sigma_yy", 1, es.syy, {}},
                                       {"tau_xy", 1, es.txy, {}},
                                       {"von_mises", 1, es.von_mises, {}}};
    if (s.plane == PlaneCondition::Strain) cell_data.push_back({"sigma_zz", 1, es.szz, {}});
    cell_data.push_back({"partition_id", 1, {}, std::vector<std::int32_t>(ne, 0)});

    std::optional<KirschMetrics> metrics;
    if (kirsch) {
      VtuArray ex{"sigma_xx_exact", 1, std::vector<double>(ne), {}};
      VtuArray ey{"sigma_yy_exact", 1, std::vector<double>(ne), {}};
      VtuArray et{"tau_xy_exact", 1, std::vector<double>(ne), {}};
      VtuArray dx{"sigma_xx_error", 1, std::vector<double>(ne), {}};
      VtuArray dy{"sigma_yy_error", 1, std::vector<double>(ne), {}};
      VtuArray dt{"tau_xy_error", 1, std::vector<double>(ne), {}};
      for (std::size_t e = 0; e < ne; ++e) {
        const Vec3 a = kirsch_stress(*kirsch, es.cx[e], es.cy[e]);
        ex.f64[e] = a[0];
        ey.f64[e] = a[1];
        et.f64[e] = a[2];
        dx.f64[e] = es.sxx[e] - a[0];
        dy.f64[e] = es.syy[e] - a[1];
        dt.f64[e] = es.txy[e] - a[2];
      }
      for (auto* a : {&ex, &ey, &et, &dx, &dy, &dt}) cell_data.push_back(std::move(*a));

      metrics = kirsch_metrics(es, *kirsch);
      write_hole_csv((dir / "hole_sigma_tt.csv").string(), mesh, es, *kirsch);
      write_section_x0_csv((dir / "section_x0_sigma_xx.csv").string(), mesh, es, *kirsch);
      out << "Kirsch: relative L2 stress error " << metrics->l2_rel_stress_error
          << "; SCF at centroid (" << metrics->scf_x << ", " << metrics->scf_y
          << "): numerical " << metrics->scf_numerical << ", analytical "
          << metrics->scf_analytical_centroid << ", theoretical 3\n";
    }

    write_vtu_piece((dir / "solution_0.vtu").string(), mesh.xy, mesh.triangles, point_data,
                    cell_data);
    write_pvtu((dir / "solution.pvtu").string(), {"solution_0.vtu"}, array_infos(point_data),
               array_infos(cell_data));
    const double t_output = seconds_since(t0);
    const double t_all = seconds_since(t_total);

    // ---- timings.json
    std::ofstream js(dir / "timings.json");
    if (!js) throw std::runtime_error((dir / "timings.json").string() + ": cannot open");
    js << "{\n"
       << "  \"case\": " << str(case_path) << ",\n"
       << "  \"mesh\": " << str(s.mesh_path) << ",\n"
       << "  \"ranks\": 1,\n"
       << "  \"plane\": " << str(s.plane_name) << ",\n"
       << "  \"nodes\": " << mesh.num_nodes() << ",\n"
       << "  \"elements\": " << ne << ",\n"
       << "  \"dofs\": " << 2 * mesh.num_nodes() << ",\n"
       << "  \"nnz\": " << K.nnz() << ",\n"
       << "  \"solver\": {\n"
       << "    \"name\": \"cg_jacobi\",\n"
       << "    \"rtol\": " << num(s.solver.rtol) << ",\n"
       << "    \"max_iter\": " << s.solver.max_iter << ",\n"
       << "    \"iterations\": " << st.iterations << ",\n"
       << "    \"rel_residual\": " << num(st.rel_residual) << ",\n"
       << "    \"true_rel_residual\": " << num(st.true_rel_residual) << ",\n"
       << "    \"converged\": " << (st.converged ? "true" : "false") << "\n"
       << "  },\n"
       << "  \"timings\": {\n"
       << "    \"read\": " << num(t_read) << ",\n"
       << "    \"assembly\": " << num(t_assembly) << ",\n"
       << "    \"bcs\": " << num(t_bcs) << ",\n"
       << "    \"solve\": " << num(t_solve) << ",\n"
       << "    \"output\": " << num(t_output) << ",\n"
       << "    \"total\": " << num(t_all) << "\n"
       << "  },\n"
       << "  \"kirsch\": ";
    if (metrics) {
      js << "{\n"
         << "    \"sigma_inf\": " << num(kirsch->sigma_inf) << ",\n"
         << "    \"a\": " << num(kirsch->a) << ",\n"
         << "    \"l2_rel_stress_error\": " << num(metrics->l2_rel_stress_error) << ",\n"
         << "    \"scf\": {\n"
         << "      \"element\": " << metrics->scf_element << ",\n"
         << "      \"x\": " << num(metrics->scf_x) << ",\n"
         << "      \"y\": " << num(metrics->scf_y) << ",\n"
         << "      \"numerical\": " << num(metrics->scf_numerical) << ",\n"
         << "      \"analytical_at_centroid\": " << num(metrics->scf_analytical_centroid) << ",\n"
         << "      \"theoretical\": " << num(metrics->scf_theoretical) << "\n"
         << "    }\n"
         << "  }\n";
    } else {
      js << "null\n";
    }
    js << "}\n";
    if (!js) throw std::runtime_error((dir / "timings.json").string() + ": write failed");

    out << "wrote " << (dir / "solution.pvtu").string() << " (1 piece), timings.json"
        << (kirsch ? ", hole_sigma_tt.csv, section_x0_sigma_xx.csv" : "") << "\n"
        << "time: read " << t_read << " s, assembly " << t_assembly << " s, bcs " << t_bcs
        << " s, solve " << t_solve << " s, output " << t_output << " s, total " << t_all
        << " s\n";
    return st.converged ? 0 : 2;
  } catch (const std::exception& e) {
    err << "error: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace cstfem
