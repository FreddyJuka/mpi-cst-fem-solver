#include "cstfem/distributed.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#include "cstfem/assembly.hpp"
#include "cstfem/mpi_util.hpp"

namespace cstfem {

namespace {

int owner_of(const std::vector<int>& offsets, int global) {
  return static_cast<int>(std::upper_bound(offsets.begin(), offsets.end(), global) -
                          offsets.begin()) -
         1;
}

struct Entry {
  int row;  // global row
  int col;  // global column
  double val;
};

}  // namespace

DofLayout make_dof_layout(const DistributedMesh& dm, MPI_Comm comm) {
  DofLayout L;
  L.offsets.resize(dm.offsets.size());
  for (std::size_t r = 0; r < dm.offsets.size(); ++r) L.offsets[r] = 2 * dm.offsets[r];
  L.n_owned = 2 * dm.num_owned;
  L.first = 2 * dm.offsets[dm.rank];
  L.n_local = 2 * dm.num_local_nodes();
  std::vector<int> ghosts;
  for (int n = dm.num_owned; n < dm.num_local_nodes(); ++n) {
    ghosts.push_back(2 * dm.local_to_global[n]);
    ghosts.push_back(2 * dm.local_to_global[n] + 1);
  }
  L.element_ghosts = std::make_unique<GhostExchange>(std::move(ghosts), L.offsets, comm);
  return L;
}

DistMatrix assemble_distributed(const DistributedMesh& dm, const DofLayout& layout,
                                const Mat3& D, double thickness, MPI_Comm comm) {
  int size = 1;
  MPI_Comm_size(comm, &size);
  DistMatrix A;
  A.n_owned = layout.n_owned;
  A.first_row = layout.first;
  A.offsets = layout.offsets;

  // COO accumulation. Memory: 36 entries (2 ints + 1 double, 16 bytes) per local triangle
  // plus the entries received from other ranks, i.e. about 576 bytes per triangle before
  // duplicate reduction; acceptable since assembly happens once (v1).
  std::vector<Entry> mine;
  std::vector<std::vector<Entry>> outgoing(size);
  mine.reserve(36 * dm.local.num_triangles());
  for (std::size_t e = 0; e < dm.local.num_triangles(); ++e) {
    const Mat6 Ke = cst_stiffness(element_coords(dm.local, e), D, thickness);
    int g[6];
    for (int k = 0; k < 3; ++k) {
      const int node = dm.local_to_global[dm.local.triangles[3 * e + k]];
      g[2 * k] = 2 * node;
      g[2 * k + 1] = 2 * node + 1;
    }
    for (int i = 0; i < 6; ++i) {
      const bool owned = g[i] >= A.first_row && g[i] < A.first_row + A.n_owned;
      auto& dest = owned ? mine : outgoing[owner_of(A.offsets, g[i])];
      for (int j = 0; j < 6; ++j) dest.push_back({g[i], g[j], Ke[i * 6 + j]});
    }
  }

  // Send contributions to rows owned elsewhere (indices and values in separate messages).
  std::vector<int> scount(size), sdispl(size), rcount(size), rdispl(size);
  std::vector<int> sidx;
  std::vector<double> sval;
  for (int q = 0; q < size; ++q) {
    sdispl[q] = static_cast<int>(sval.size());
    scount[q] = static_cast<int>(outgoing[q].size());
    for (const Entry& en : outgoing[q]) {
      sidx.push_back(en.row);
      sidx.push_back(en.col);
      sval.push_back(en.val);
    }
  }
  MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, comm);
  int nrecv = 0;
  for (int q = 0; q < size; ++q) {
    rdispl[q] = nrecv;
    nrecv += rcount[q];
  }
  std::vector<double> rval(nrecv);
  std::vector<int> ridx(2 * static_cast<std::size_t>(nrecv));
  MPI_Alltoallv(sval.data(), scount.data(), sdispl.data(), MPI_DOUBLE, rval.data(),
                rcount.data(), rdispl.data(), MPI_DOUBLE, comm);
  for (int q = 0; q < size; ++q) {
    scount[q] *= 2;
    sdispl[q] *= 2;
    rcount[q] *= 2;
    rdispl[q] *= 2;
  }
  MPI_Alltoallv(sidx.data(), scount.data(), sdispl.data(), MPI_INT, ridx.data(), rcount.data(),
                rdispl.data(), MPI_INT, comm);
  for (int i = 0; i < nrecv; ++i) mine.push_back({ridx[2 * i], ridx[2 * i + 1], rval[i]});

  // Sort by (row, column) and sum duplicates.
  std::stable_sort(mine.begin(), mine.end(), [](const Entry& a, const Entry& b) {
    return a.row != b.row ? a.row < b.row : a.col < b.col;
  });
  std::vector<Entry> reduced;
  for (const Entry& en : mine) {
    if (!reduced.empty() && reduced.back().row == en.row && reduced.back().col == en.col)
      reduced.back().val += en.val;
    else
      reduced.push_back(en);
  }

  // SpMV halo: non-owned columns of the owned rows.
  const int lo = A.first_row, hi = A.first_row + A.n_owned;
  for (const Entry& en : reduced)
    if (en.col < lo || en.col >= hi) A.halo_globals.push_back(en.col);
  std::sort(A.halo_globals.begin(), A.halo_globals.end());
  A.halo_globals.erase(std::unique(A.halo_globals.begin(), A.halo_globals.end()),
                       A.halo_globals.end());

  // Split into A_loc (owned columns) and A_ghost (halo columns), local column numbering.
  for (CsrMatrix* M : {&A.loc, &A.ghost}) {
    M->num_rows = A.n_owned;
    M->first_row = 0;
    M->row_ptr.assign(A.n_owned + 1, 0);
  }
  A.loc.num_cols = A.n_owned;
  A.ghost.num_cols = A.n_halo();
  for (const Entry& en : reduced) {
    if (en.row < lo || en.row >= hi) throw std::logic_error("assemble_distributed: foreign row");
    const int r = en.row - lo;
    if (en.col >= lo && en.col < hi) {
      ++A.loc.row_ptr[r + 1];
      A.loc.cols.push_back(en.col - lo);
      A.loc.vals.push_back(en.val);
    } else {
      ++A.ghost.row_ptr[r + 1];
      A.ghost.cols.push_back(static_cast<int>(
          std::lower_bound(A.halo_globals.begin(), A.halo_globals.end(), en.col) -
          A.halo_globals.begin()));
      A.ghost.vals.push_back(en.val);
    }
  }
  for (CsrMatrix* M : {&A.loc, &A.ghost})
    for (int r = 0; r < A.n_owned; ++r) M->row_ptr[r + 1] += M->row_ptr[r];

  A.halo = std::make_unique<GhostExchange>(A.halo_globals, A.offsets, comm);
  return A;
}

std::vector<double> accumulate_owned(DofLayout& layout, const std::vector<double>& f_local) {
  std::vector<double> b(f_local.begin(), f_local.begin() + layout.n_owned);
  layout.element_ghosts->reverse_add(f_local.data() + layout.n_owned, b.data());
  return b;
}

std::vector<double> to_mesh_local(DofLayout& layout, const std::vector<double>& u_owned) {
  std::vector<double> u(layout.n_local, 0.0);
  std::copy(u_owned.begin(), u_owned.end(), u.begin());
  layout.element_ghosts->forward(u_owned.data(), u.data() + layout.n_owned);
  return u;
}

void apply_dirichlet_distributed(const DistributedMesh& dm, DofLayout& layout,
                                 const DirichletDofs& local, DistMatrix& A,
                                 std::vector<double>& b, MPI_Comm comm) {
  const int n_owned = layout.n_owned, n_ghost = layout.n_local - layout.n_owned;

  // (a) Own prescriptions + those found by element-ghost holders, combined at the owner.
  std::vector<double> ghost_flag(n_ghost, 0.0), ghost_value(n_ghost, 0.0);
  std::vector<char> fixed(n_owned, 0);
  std::vector<double> value(n_owned, 0.0);
  for (const auto& [dof, v] : local.values()) {
    if (dof < n_owned) {
      fixed[dof] = 1;
      value[dof] = v;
    } else {
      ghost_flag[dof - n_owned] = 1.0;
      ghost_value[dof - n_owned] = v;
    }
  }
  const std::vector<double> flags_in = layout.element_ghosts->reverse(ghost_flag.data());
  const std::vector<double> values_in = layout.element_ghosts->reverse(ghost_value.data());
  const auto& target = layout.element_ghosts->reverse_targets();
  bool ok = true;
  std::string error;
  for (std::size_t i = 0; i < flags_in.size() && ok; ++i) {
    if (flags_in[i] == 0.0) continue;
    const int t = target[i];
    if (fixed[t] && value[t] != values_in[i]) {
      std::ostringstream msg;
      msg.precision(17);
      msg << "conflicting Dirichlet values for node tag " << dm.local.node_tags[t / 2]
          << " component " << (t % 2 == 0 ? "ux" : "uy") << ": " << value[t] << " vs "
          << values_in[i];
      ok = false;
      error = msg.str();
    }
    fixed[t] = 1;
    value[t] = values_in[i];
  }
  check_collective(ok, error, comm);

  // (b) Constraint status of the halo columns, from their owners.
  std::vector<double> owned_flag(n_owned), halo_flag(A.n_halo(), 0.0), halo_value(A.n_halo(), 0.0);
  for (int i = 0; i < n_owned; ++i) owned_flag[i] = fixed[i];
  A.halo->forward(owned_flag.data(), halo_flag.data());
  A.halo->forward(value.data(), halo_value.data());

  // (c) Elimination: owned block with the serial routine, then the halo columns.
  DirichletDofs owned;
  for (int i = 0; i < n_owned; ++i)
    if (fixed[i]) owned.add(i, value[i]);
  apply_dirichlet(owned, A.loc, b);
  CsrMatrix& G = A.ghost;
  for (int i = 0; i < n_owned; ++i) {
    for (int k = G.row_ptr[i]; k < G.row_ptr[i + 1]; ++k) {
      if (fixed[i]) {
        G.vals[k] = 0.0;  // constrained row: only the diagonal (in A_loc) survives
      } else if (halo_flag[G.cols[k]] != 0.0) {
        b[i] -= G.vals[k] * halo_value[G.cols[k]];
        G.vals[k] = 0.0;
      }
    }
  }
}

void DistOperator::apply(const std::vector<double>& x, std::vector<double>& y) const {
  A_.halo->begin_forward(x.data());
  A_.loc.multiply(x, y);  // overlaps the halo exchange
  A_.halo->end_forward(halo_values_.data());
  const CsrMatrix& G = A_.ghost;
  for (int i = 0; i < G.num_rows; ++i) {
    double s = 0.0;
    for (int k = G.row_ptr[i]; k < G.row_ptr[i + 1]; ++k) s += G.vals[k] * halo_values_[G.cols[k]];
    y[i] += s;
  }
}

double DistReduction::dot(const std::vector<double>& a, const std::vector<double>& b) const {
  double local = 0.0, global = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) local += a[i] * b[i];
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm_);
  return global;
}

SolverStats solve_distributed(const DistMatrix& A, const std::vector<double>& b,
                              std::vector<double>& x, const SolverOptions& options,
                              MPI_Comm comm) {
  std::vector<double> inv_diag(A.n_owned);
  bool ok = true;
  std::string error;
  for (int i = 0; i < A.n_owned && ok; ++i) {
    const double d = A.loc.diagonal(i);
    if (!(d > 0.0)) {
      ok = false;
      error = "cg_jacobi: non-positive diagonal at global row " + std::to_string(A.first_row + i);
    }
    inv_diag[i] = 1.0 / d;
  }
  check_collective(ok, error, comm);
  return pcg_jacobi(DistOperator(A), inv_diag, DistReduction(comm), b, x, options);
}

KirschMetrics kirsch_metrics_distributed(const ElementStresses& s, const KirschParams& p,
                                         MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  const KirschPartial k = kirsch_partial(s, p);
  double sums[2] = {k.err2, k.ref2}, total[2] = {0.0, 0.0};
  MPI_Allreduce(sums, total, 2, MPI_DOUBLE, MPI_SUM, comm);

  struct {
    double distance;
    int rank;
  } mine{k.scf_distance, rank}, best{0.0, 0};
  MPI_Allreduce(&mine, &best, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);

  KirschMetrics m;
  m.l2_rel_stress_error = std::sqrt(total[0] / total[1]);
  if (rank == best.rank && k.scf_element >= 0) set_scf(m, s, p, k.scf_element);
  double v[5] = {m.scf_x, m.scf_y, m.scf_numerical, m.scf_analytical_centroid,
                 static_cast<double>(m.scf_element)};
  MPI_Bcast(v, 5, MPI_DOUBLE, best.rank, comm);
  m.scf_x = v[0];
  m.scf_y = v[1];
  m.scf_numerical = v[2];
  m.scf_analytical_centroid = v[3];
  m.scf_element = static_cast<int>(v[4]);  // local index on the winning rank
  return m;
}

}  // namespace cstfem
