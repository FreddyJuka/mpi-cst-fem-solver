#pragma once

#include <mpi.h>

#include <memory>
#include <vector>

#include "cstfem/bc.hpp"
#include "cstfem/csr.hpp"
#include "cstfem/element.hpp"
#include "cstfem/ghost_exchange.hpp"
#include "cstfem/partition.hpp"
#include "cstfem/postprocess.hpp"
#include "cstfem/solver.hpp"

namespace cstfem {

// DOF view of a DistributedMesh. Mesh-local DOF of local node n, component c: 2 n + c, so
// owned DOFs come first, [0, n_owned), followed by the element-ghost DOFs.
struct DofLayout {
  std::vector<int> offsets;  // global DOF offsets (2 x node offsets)
  int n_owned = 0;           // owned DOFs; global id of owned DOF i is first + i
  int first = 0;
  int n_local = 0;           // owned + element-ghost DOFs
  // Owner <-> element-ghost DOF copies (the element-ghost set of 4a, in DOF units):
  // used for loads (reverse) and for nodal displacements of local triangles (forward).
  std::unique_ptr<GhostExchange> element_ghosts;
};

DofLayout make_dof_layout(const DistributedMesh& dm, MPI_Comm comm);

// Row-owner distributed matrix (SPEC section 5). Rows: owned DOFs. Columns are renumbered
// locally: owned [0, n_owned) in `loc`, SpMV halo [0, n_halo) in `ghost`, where halo column k
// is global DOF halo_globals[k]. The halo is the set of non-owned global columns of the owned
// rows; it differs from the element-ghost set (see DistributedMesh).
struct DistMatrix {
  int n_owned = 0;
  int first_row = 0;
  std::vector<int> offsets;       // global DOF offsets
  std::vector<int> halo_globals;  // sorted
  CsrMatrix loc;                  // n_owned x n_owned
  CsrMatrix ghost;                // n_owned x n_halo
  std::unique_ptr<GhostExchange> halo;

  int n_halo() const { return static_cast<int>(halo_globals.size()); }
  std::size_t nnz() const { return loc.nnz() + ghost.nnz(); }
};

// Assembles the local triangles; contributions to rows owned elsewhere are sent to their
// owner (MPI_Alltoallv) and summed there. Builds the halo exchange. Collective.
DistMatrix assemble_distributed(const DistributedMesh& dm, const DofLayout& layout,
                                const Mat3& D, double thickness, MPI_Comm comm);

// Owned right-hand side from a mesh-local load vector (owned + element-ghost DOFs):
// element-ghost contributions are reverse-accumulated to their owners. Collective.
std::vector<double> accumulate_owned(DofLayout& layout, const std::vector<double>& f_local);

// Owned solution -> mesh-local vector (owned + element ghosts, forward exchange). Collective.
std::vector<double> to_mesh_local(DofLayout& layout, const std::vector<double>& u_owned);

// Distributed Dirichlet elimination (SPEC section 6). `local` holds the prescriptions this
// rank found on its own boundary edges, keyed by mesh-local DOF (owned or element ghost).
// (a) ghost prescriptions are reverse-accumulated to the owners (conflicting values: error),
// (b) owned flags/values are forwarded over the SpMV halo,
// (c) symmetric elimination on owned rows and on owned + halo columns, keeping K_jj and
//     setting b_j = K_jj u_j. Collective.
void apply_dirichlet_distributed(const DistributedMesh& dm, DofLayout& layout,
                                 const DirichletDofs& local, DistMatrix& A,
                                 std::vector<double>& b, MPI_Comm comm);

// y = A x with x, y over owned DOFs: start halo exchange, y = A_loc x, wait,
// y += A_ghost x_halo.
class DistOperator : public LinearOperator {
public:
  explicit DistOperator(const DistMatrix& A) : A_(A), halo_values_(A.n_halo()) {}
  void apply(const std::vector<double>& x, std::vector<double>& y) const override;

private:
  const DistMatrix& A_;
  mutable std::vector<double> halo_values_;
};

// Local dot product + MPI_Allreduce.
class DistReduction : public Reduction {
public:
  explicit DistReduction(MPI_Comm comm) : comm_(comm) {}
  double dot(const std::vector<double>& a, const std::vector<double>& b) const override;

private:
  MPI_Comm comm_;
};

// Jacobi-PCG of cstfem_core with the distributed operator and inner product. Collective.
SolverStats solve_distributed(const DistMatrix& A, const std::vector<double>& b,
                              std::vector<double>& x, const SolverOptions& options,
                              MPI_Comm comm);

// Kirsch metrics over all ranks: L2 sums via MPI_Allreduce, SCF element via MPI_MINLOC on
// the centroid distance to (0, a), values broadcast from the winning rank. Collective.
KirschMetrics kirsch_metrics_distributed(const ElementStresses& s, const KirschParams& p,
                                         MPI_Comm comm);

}  // namespace cstfem
