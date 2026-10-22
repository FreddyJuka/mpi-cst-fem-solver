#include "cstfem/parallel_driver.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "cstfem/case.hpp"
#include "cstfem/distributed.hpp"
#include "cstfem/driver.hpp"
#include "cstfem/json.hpp"
#include "cstfem/mpi_util.hpp"
#include "cstfem/partition.hpp"
#include "cstfem/vtu.hpp"

namespace cstfem {

namespace {

// Rank 0 reads the case file; its text is broadcast and parsed on every rank.
CaseFile load_case_collective(const std::string& case_path, MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  std::string text, error;
  bool ok = true;
  if (rank == 0) {
    std::ifstream in(case_path);
    if (!in) {
      ok = false;
      error = case_path + ": cannot open file";
    } else {
      std::ostringstream ss;
      ss << in.rdbuf();
      text = ss.str();
    }
  }
  check_collective(ok, error, comm);
  text = bcast_string(text, 0, comm);
  std::istringstream in(text);
  return CaseFile::parse(in, case_path);  // same text everywhere: errors are collective
}

// Runs `f` (local work only, no collectives) and turns an exception on any rank into an
// exception on every rank.
template <class F>
void collective(MPI_Comm comm, F&& f) {
  bool ok = true;
  std::string error;
  try {
    f();
  } catch (const std::exception& e) {
    ok = false;
    error = e.what();
  }
  check_collective(ok, error, comm);
}

Partitioner to_partitioner(const std::string& name) {
  return name == "parmetis" ? Partitioner::ParMetis : Partitioner::Metis;
}

// Per-rank wall time of each phase; every phase starts after an MPI_Barrier.
class PhaseTimer {
public:
  explicit PhaseTimer(MPI_Comm comm) : comm_(comm) {}
  void start() {
    MPI_Barrier(comm_);
    t0_ = MPI_Wtime();
  }
  void stop(const std::string& phase) { set(phase, MPI_Wtime() - t0_); }
  void set(const std::string& phase, double seconds) {
    if (!times_.count(phase)) order_.push_back(phase);
    times_[phase] = seconds;
  }
  // (min, max, mean) over ranks per phase, valid on rank 0.
  std::vector<std::array<double, 3>> stats() const {
    int nranks = 1;
    MPI_Comm_size(comm_, &nranks);
    const std::size_t n = order_.size();
    std::vector<double> mine(n), mn(n), mx(n), sum(n);
    for (std::size_t i = 0; i < n; ++i) mine[i] = times_.at(order_[i]);
    MPI_Reduce(mine.data(), mn.data(), static_cast<int>(n), MPI_DOUBLE, MPI_MIN, 0, comm_);
    MPI_Reduce(mine.data(), mx.data(), static_cast<int>(n), MPI_DOUBLE, MPI_MAX, 0, comm_);
    MPI_Reduce(mine.data(), sum.data(), static_cast<int>(n), MPI_DOUBLE, MPI_SUM, 0, comm_);
    std::vector<std::array<double, 3>> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = {mn[i], mx[i], sum[i] / nranks};
    return out;
  }
  const std::vector<std::string>& phases() const { return order_; }

private:
  MPI_Comm comm_;
  double t0_ = 0.0;
  std::map<std::string, double> times_;
  std::vector<std::string> order_;
};

// Per-rank load figures gathered on rank 0: owned rows, halo size, neighbors, local nnz.
struct LoadReport {
  std::vector<std::array<long long, 4>> ranks;
  double rows_max_over_mean = 0.0, nnz_max_over_mean = 0.0;
  long long global_nnz = 0;
};

LoadReport gather_load(const DistMatrix& A, MPI_Comm comm) {
  int rank = 0, nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);
  const long long mine[4] = {A.n_owned, A.n_halo(), A.halo->num_neighbors(),
                             static_cast<long long>(A.nnz())};
  std::vector<long long> all(rank == 0 ? 4 * nranks : 0);
  MPI_Gather(mine, 4, MPI_LONG_LONG, all.data(), 4, MPI_LONG_LONG, 0, comm);
  LoadReport L;
  if (rank != 0) return L;
  long long max_rows = 0, max_nnz = 0, sum_rows = 0;
  for (int r = 0; r < nranks; ++r) {
    std::array<long long, 4> v;
    for (int k = 0; k < 4; ++k) v[k] = all[4 * static_cast<std::size_t>(r) + k];
    L.ranks.push_back(v);
    max_rows = std::max(max_rows, v[0]);
    max_nnz = std::max(max_nnz, v[3]);
    sum_rows += v[0];
    L.global_nnz += v[3];
  }
  L.rows_max_over_mean = max_rows / (static_cast<double>(sum_rows) / nranks);
  L.nnz_max_over_mean = max_nnz / (static_cast<double>(L.global_nnz) / nranks);
  return L;
}

void print_load_report(const LoadReport& L, std::ostream& out) {
  out << "Load report:\n"
      << std::setw(5) << "rank" << std::setw(12) << "owned rows" << std::setw(8) << "halo"
      << std::setw(11) << "neighbors" << std::setw(12) << "local nnz" << "\n";
  for (std::size_t r = 0; r < L.ranks.size(); ++r)
    out << std::setw(5) << r << std::setw(12) << L.ranks[r][0] << std::setw(8) << L.ranks[r][1]
        << std::setw(11) << L.ranks[r][2] << std::setw(12) << L.ranks[r][3] << "\n";
  out << std::fixed << std::setprecision(4) << "owned rows max/mean " << L.rows_max_over_mean
      << ", local nnz max/mean " << L.nnz_max_over_mean << std::defaultfloat << "\n";
}

// Gathers flattened CSV rows on rank 0.
std::vector<double> gather_rows(const std::vector<double>& mine, MPI_Comm comm) {
  int rank = 0, nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);
  int n = static_cast<int>(mine.size());
  std::vector<int> counts(nranks), displs(nranks);
  MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
  int total = 0;
  for (int r = 0; r < nranks; ++r) {
    displs[r] = total;
    total += counts[r];
  }
  std::vector<double> all(rank == 0 ? total : 0);
  MPI_Gatherv(mine.data(), n, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE,
              0, comm);
  return all;
}

int run_distributed(const CaseFile& c, const CaseSettings& s, const std::string& case_path,
                    MPI_Comm comm, std::ostream& out, std::ostream& err) {
  int rank = 0, nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);
  if (rank == 0) c.warn_unknown_keys(known_case_keys(), err);
  PhaseTimer timer(comm);
  MPI_Barrier(comm);
  const double t_begin = MPI_Wtime();

  // ---- read, partition, distribute (phases timed inside)
  PartitionSummary summary;
  PartitionTimings pt;
  const DistributedMesh dm = read_partition_distribute(
      s.mesh_path, comm, &summary, to_partitioner(s.partitioner), kMetisSeed, &pt);
  timer.set("read", pt.read);
  timer.set("partition", pt.partition);
  timer.set("distribute", pt.distribute);
  if (rank == 0)
    out << "mesh " << s.mesh_path << ": " << dm.global_num_nodes << " nodes, "
        << dm.global_num_triangles << " triangles, " << 2 * dm.global_num_nodes << " DOFs, "
        << nranks << " rank(s), distributed path, partitioner " << summary.partitioner
        << " (edge cut " << summary.edge_cut << ")\n";

  // ---- assembly
  timer.start();
  DofLayout layout = make_dof_layout(dm, comm);
  const Mat3 D = constitutive_matrix(s.E, s.nu, s.plane);
  DistMatrix A = assemble_distributed(dm, layout, D, s.thickness, comm);
  timer.stop("assembly");

  // ---- boundary conditions
  timer.start();
  std::vector<double> f_local(layout.n_local, 0.0);
  std::optional<KirschParams> kirsch;
  DirichletDofs local_dofs;
  collective(comm, [&] {
    kirsch = add_case_tractions(c, dm.local, s.thickness, f_local);
    add_dirichlet(dm.local, parse_dirichlet(c.get_string("bc.dirichlet", "")), local_dofs);
  });
  std::vector<double> b = accumulate_owned(layout, f_local);
  apply_dirichlet_distributed(dm, layout, local_dofs, A, b, comm);
  timer.stop("bcs");

  // ---- solve
  timer.start();
  std::vector<double> u;
  const SolverStats st = solve_distributed(A, b, u, s.solver, comm);
  timer.stop("solve");
  if (rank == 0) {
    out << "cg_jacobi: " << st.message << "; true relative residual " << st.true_rel_residual
        << "\n";
    if (!st.warning.empty()) err << st.warning << "\n";
    if (!st.converged) err << "error: solver did not converge: " << st.message << "\n";
  }

  // ---- postprocess: element-ghost displacements, element stresses, metrics
  timer.start();
  const std::vector<double> u_local = to_mesh_local(layout, u);
  const ElementStresses es = compute_element_stresses(dm.local, u_local, D, s.nu, s.plane);
  std::optional<KirschMetrics> metrics;
  if (kirsch) metrics = kirsch_metrics_distributed(es, *kirsch, comm);
  timer.stop("postprocess");

  // ---- output: one VTU piece per rank, .pvtu and gathered CSVs on rank 0
  timer.start();
  const std::filesystem::path dir(s.output_dir);
  collective(comm, [&] {
    if (rank == 0) {
      std::filesystem::create_directories(dir);
      remove_stale_pieces(s.output_dir, "solution", nranks);
    }
  });
  const SolutionPiece piece = solution_arrays(dm.local, u_local, es, s.plane, dm.num_owned,
                                              rank, dm.element_ids, kirsch);
  collective(comm, [&] {
    write_vtu_piece((dir / ("solution_" + std::to_string(rank) + ".vtu")).string(), dm.local.xy,
                    dm.local.triangles, piece.point_data, piece.cell_data);
  });
  std::vector<double> hole, section;
  if (kirsch) {
    hole = gather_rows(hole_csv_rows(dm.local, es, *kirsch), comm);
    section = gather_rows(section_x0_csv_rows(dm.local, es, *kirsch), comm);
  }
  collective(comm, [&] {
    if (rank != 0) return;
    std::vector<std::string> pieces;
    for (int r = 0; r < nranks; ++r) pieces.push_back("solution_" + std::to_string(r) + ".vtu");
    write_pvtu((dir / "solution.pvtu").string(), pieces, array_infos(piece.point_data),
               array_infos(piece.cell_data));
    if (kirsch) {
      write_hole_csv((dir / "hole_sigma_tt.csv").string(), std::move(hole));
      write_section_x0_csv((dir / "section_x0_sigma_xx.csv").string(), std::move(section));
    }
  });
  timer.stop("output");
  MPI_Barrier(comm);
  timer.set("total", MPI_Wtime() - t_begin);

  // ---- reports and timings.json (rank 0)
  const auto stats = timer.stats();
  const LoadReport load = gather_load(A, comm);
  collective(comm, [&] {
    if (rank != 0) return;
    if (metrics)
      out << "Kirsch: relative L2 stress error " << metrics->l2_rel_stress_error
          << "; SCF at centroid (" << metrics->scf_x << ", " << metrics->scf_y
          << "): numerical " << metrics->scf_numerical << ", analytical "
          << metrics->scf_analytical_centroid << ", theoretical 3\n";
    print_load_report(load, out);
    out << "time per phase [s] (min / max / mean over ranks):\n";
    for (std::size_t i = 0; i < stats.size(); ++i)
      out << "  " << std::left << std::setw(11) << timer.phases()[i] << std::right
          << std::setw(11) << std::setprecision(4) << stats[i][0] << std::setw(11) << stats[i][1]
          << std::setw(11) << stats[i][2] << "\n";
    out << std::setprecision(6);

    std::ofstream js(dir / "timings.json");
    if (!js) throw std::runtime_error((dir / "timings.json").string() + ": cannot open");
    auto list = [&](int k) {
      std::string l = "[";
      for (std::size_t r = 0; r < load.ranks.size(); ++r)
        l += (r ? ", " : "") + std::to_string(load.ranks[r][k]);
      return l + "]";
    };
    js << "{\n"
       << "  \"case\": " << json_string(case_path) << ",\n"
       << "  \"mesh\": " << json_string(s.mesh_path) << ",\n"
       << "  \"ranks\": " << nranks << ",\n"
       << "  \"path\": \"distributed\",\n"
       << "  \"partitioner\": " << json_string(summary.partitioner) << ",\n"
       << "  \"plane\": " << json_string(s.plane_name) << ",\n"
       << "  \"nodes\": " << dm.global_num_nodes << ",\n"
       << "  \"elements\": " << dm.global_num_triangles << ",\n"
       << "  \"dofs\": " << 2 * dm.global_num_nodes << ",\n"
       << "  \"nnz\": " << load.global_nnz << ",\n"
       << "  \"partition\": {\"edge_cut\": " << summary.edge_cut
       << ", \"interface_nodes\": " << summary.interface_nodes << "},\n"
       << "  \"solver\": {\n"
       << "    \"name\": \"cg_jacobi\",\n"
       << "    \"rtol\": " << json_number(s.solver.rtol) << ",\n"
       << "    \"max_iter\": " << s.solver.max_iter << ",\n"
       << "    \"iterations\": " << st.iterations << ",\n"
       << "    \"rel_residual\": " << json_number(st.rel_residual) << ",\n"
       << "    \"true_rel_residual\": " << json_number(st.true_rel_residual) << ",\n"
       << "    \"converged\": " << (st.converged ? "true" : "false") << "\n"
       << "  },\n"
       << "  \"timings\": {\n";
    for (std::size_t i = 0; i < stats.size(); ++i)
      js << "    " << json_string(timer.phases()[i]) << ": {\"min\": " << json_number(stats[i][0])
         << ", \"max\": " << json_number(stats[i][1]) << ", \"mean\": "
         << json_number(stats[i][2]) << "}" << (i + 1 < stats.size() ? "," : "") << "\n";
    js << "  },\n"
       << "  \"load\": {\n"
       << "    \"owned_rows\": " << list(0) << ",\n"
       << "    \"halo\": " << list(1) << ",\n"
       << "    \"neighbors\": " << list(2) << ",\n"
       << "    \"local_nnz\": " << list(3) << ",\n"
       << "    \"owned_rows_max_over_mean\": " << json_number(load.rows_max_over_mean) << ",\n"
       << "    \"local_nnz_max_over_mean\": " << json_number(load.nnz_max_over_mean) << "\n"
       << "  },\n"
       << "  \"kirsch\": ";
    if (metrics) {
      js << "{\n"
         << "    \"sigma_inf\": " << json_number(kirsch->sigma_inf) << ",\n"
         << "    \"a\": " << json_number(kirsch->a) << ",\n"
         << "    \"l2_rel_stress_error\": " << json_number(metrics->l2_rel_stress_error) << ",\n"
         << "    \"scf\": {\n"
         << "      \"x\": " << json_number(metrics->scf_x) << ",\n"
         << "      \"y\": " << json_number(metrics->scf_y) << ",\n"
         << "      \"numerical\": " << json_number(metrics->scf_numerical) << ",\n"
         << "      \"analytical_at_centroid\": " << json_number(metrics->scf_analytical_centroid)
         << ",\n"
         << "      \"theoretical\": " << json_number(metrics->scf_theoretical) << "\n"
         << "    }\n"
         << "  }\n";
    } else {
      js << "null\n";
    }
    js << "}\n";
    if (!js) throw std::runtime_error((dir / "timings.json").string() + ": write failed");
    out << "wrote " << (dir / "solution.pvtu").string() << " (" << nranks << " piece(s)), "
        << "timings.json" << (kirsch ? ", hole_sigma_tt.csv, section_x0_sigma_xx.csv" : "")
        << "\n";
  });
  return st.converged ? 0 : 2;
}

}  // namespace

int run_partition_only(const std::string& case_path, MPI_Comm comm, std::ostream& out,
                       std::ostream& err) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  try {
    const CaseFile c = load_case_collective(case_path, comm);
    if (rank == 0) c.warn_unknown_keys(known_case_keys(), err);
    std::string mesh_path, output_dir, partitioner;
    collective(comm, [&] {
      mesh_path = c.get_string("mesh");
      output_dir = c.get_string("output_dir", "results");
      partitioner = c.get_string("partitioner", "metis");
      if (partitioner != "metis" && partitioner != "parmetis")
        throw std::runtime_error(c.source() + ": partitioner must be metis or parmetis");
    });
    PartitionSummary summary;
    const DistributedMesh dm =
        read_partition_distribute(mesh_path, comm, &summary, to_partitioner(partitioner));
    print_partition_report(dm, summary, comm, out);
    write_partition_vtu(dm, output_dir, "partition", comm);
    if (rank == 0)
      out << "wrote " << output_dir << "/partition.pvtu (" << dm.nranks << " piece(s))\n";
    return 0;
  } catch (const std::exception& e) {
    // Errors are collective (check_collective), so every rank gets here together.
    if (rank == 0) err << "error: " << e.what() << "\n";
    return 1;
  }
}

int run_case(const std::string& case_path, MPI_Comm comm, std::ostream& out, std::ostream& err) {
  int rank = 0, nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);
  std::string path;
  try {
    const CaseFile c = load_case_collective(case_path, comm);
    CaseSettings s;
    collective(comm, [&] {
      s = read_case_settings(c);
      path = resolve_path(s, nranks);
    });
    if (path == "distributed") return run_distributed(c, s, case_path, comm, out, err);
  } catch (const std::exception& e) {
    if (rank == 0) err << "error: " << e.what() << "\n";
    return 1;
  }
  // Serial path (1 rank): unchanged serial driver.
  return run_serial_case(case_path, out, err);
}

}  // namespace cstfem
