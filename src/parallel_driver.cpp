#include "cstfem/parallel_driver.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "cstfem/case.hpp"
#include "cstfem/distributed.hpp"
#include "cstfem/driver.hpp"
#include "cstfem/mpi_util.hpp"
#include "cstfem/partition.hpp"

namespace cstfem {

namespace {

// Rank 0 reads the case file; its text is broadcast and parsed on every rank.
CaseFile load_case_collective(const std::string& case_path, MPI_Comm comm, std::ostream& err) {
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
  CaseFile c = CaseFile::parse(in, case_path);  // same text everywhere: errors are collective
  if (rank == 0) c.warn_unknown_keys(known_case_keys(), err);
  return c;
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

double max_over_ranks(double v, MPI_Comm comm) {
  double m = 0.0;
  MPI_Allreduce(&v, &m, 1, MPI_DOUBLE, MPI_MAX, comm);
  return m;
}

void print_load_report(const DistMatrix& A, int nranks, MPI_Comm comm, std::ostream& out) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  const long long mine[4] = {A.n_owned, A.n_halo(), A.halo->num_neighbors(),
                             static_cast<long long>(A.nnz())};
  std::vector<long long> all(rank == 0 ? 4 * nranks : 0);
  MPI_Gather(mine, 4, MPI_LONG_LONG, all.data(), 4, MPI_LONG_LONG, 0, comm);
  if (rank != 0) return;
  out << "Load report:\n"
      << std::setw(5) << "rank" << std::setw(12) << "owned rows" << std::setw(8) << "halo"
      << std::setw(11) << "neighbors" << std::setw(12) << "local nnz" << "\n";
  long long max_rows = 0, max_nnz = 0, sum_rows = 0, sum_nnz = 0;
  for (int r = 0; r < nranks; ++r) {
    const long long* v = &all[4 * static_cast<std::size_t>(r)];
    out << std::setw(5) << r << std::setw(12) << v[0] << std::setw(8) << v[1] << std::setw(11)
        << v[2] << std::setw(12) << v[3] << "\n";
    max_rows = std::max(max_rows, v[0]);
    max_nnz = std::max(max_nnz, v[3]);
    sum_rows += v[0];
    sum_nnz += v[3];
  }
  out << std::fixed << std::setprecision(4) << "owned rows max/mean "
      << max_rows / (static_cast<double>(sum_rows) / nranks) << ", local nnz max/mean "
      << max_nnz / (static_cast<double>(sum_nnz) / nranks) << std::defaultfloat << "\n";
}

}  // namespace

int run_partition_only(const std::string& case_path, MPI_Comm comm, std::ostream& out,
                       std::ostream& err) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  try {
    const CaseFile c = load_case_collective(case_path, comm, err);
    std::string mesh_path, output_dir;
    collective(comm, [&] {
      mesh_path = c.get_string("mesh");
      output_dir = c.get_string("output_dir", "results");
    });
    PartitionSummary summary;
    const DistributedMesh dm = read_partition_distribute(mesh_path, comm, &summary);
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

int run_parallel_case(const std::string& case_path, MPI_Comm comm, std::ostream& out,
                      std::ostream& err) {
  int rank = 0, nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);
  try {
    const CaseFile c = load_case_collective(case_path, comm, err);
    CaseSettings s;
    collective(comm, [&] { s = read_case_settings(c); });

    double t0 = MPI_Wtime();
    PartitionSummary summary;
    const DistributedMesh dm = read_partition_distribute(s.mesh_path, comm, &summary);
    const double t_read = max_over_ranks(MPI_Wtime() - t0, comm);
    if (rank == 0)
      out << "mesh " << s.mesh_path << ": " << dm.global_num_nodes << " nodes, "
          << dm.global_num_triangles << " triangles, " << 2 * dm.global_num_nodes
          << " DOFs, " << nranks << " ranks (METIS edge cut " << summary.edge_cut << ")\n";

    t0 = MPI_Wtime();
    DofLayout layout = make_dof_layout(dm, comm);
    const Mat3 D = constitutive_matrix(s.E, s.nu, s.plane);
    DistMatrix A = assemble_distributed(dm, layout, D, s.thickness, comm);
    const double t_assembly = max_over_ranks(MPI_Wtime() - t0, comm);

    t0 = MPI_Wtime();
    std::vector<double> f_local(layout.n_local, 0.0);
    std::optional<KirschParams> kirsch;
    DirichletDofs local_dofs;
    collective(comm, [&] {
      kirsch = add_case_tractions(c, dm.local, s.thickness, f_local);
      add_dirichlet(dm.local, parse_dirichlet(c.get_string("bc.dirichlet", "")), local_dofs);
    });
    std::vector<double> b = accumulate_owned(layout, f_local);
    apply_dirichlet_distributed(dm, layout, local_dofs, A, b, comm);
    const double t_bcs = max_over_ranks(MPI_Wtime() - t0, comm);

    t0 = MPI_Wtime();
    std::vector<double> u;
    const SolverStats st = solve_distributed(A, b, u, s.solver, comm);
    const double t_solve = max_over_ranks(MPI_Wtime() - t0, comm);
    if (rank == 0) {
      out << "cg_jacobi: " << st.message << "; true relative residual " << st.true_rel_residual
          << "\n";
      if (!st.warning.empty()) err << st.warning << "\n";
      if (!st.converged) err << "error: solver did not converge: " << st.message << "\n";
    }

    if (kirsch) {
      const std::vector<double> u_local = to_mesh_local(layout, u);
      const ElementStresses es = compute_element_stresses(dm.local, u_local, D, s.nu, s.plane);
      const KirschMetrics m = kirsch_metrics_distributed(es, *kirsch, comm);
      if (rank == 0)
        out << "Kirsch: relative L2 stress error " << m.l2_rel_stress_error
            << "; SCF at centroid (" << m.scf_x << ", " << m.scf_y << "): numerical "
            << m.scf_numerical << ", analytical " << m.scf_analytical_centroid
            << ", theoretical 3\n";
    }
    print_load_report(A, nranks, comm, out);
    if (rank == 0)
      out << "time (max over ranks): read+partition+distribute " << t_read << " s, assembly "
          << t_assembly << " s, bcs " << t_bcs << " s, solve " << t_solve << " s\n"
          << "parallel output arrives in Milestone 4c: VTU/CSV/timings.json not written\n";
    return st.converged ? 0 : 2;
  } catch (const std::exception& e) {
    if (rank == 0) err << "error: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace cstfem
