#pragma once

#include <mpi.h>

#include <map>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cstfem/mesh.hpp"

namespace cstfem {

// Fixed METIS seed so that the same mesh and rank count always give the same partition.
constexpr int kMetisSeed = 42;

// Element partition, node ownership and global numbering of a whole mesh (SPEC section 4),
// computed on one rank.
struct GlobalPartition {
  int nparts = 1;
  std::vector<int> elem_part;   // triangle -> rank
  std::vector<int> node_owner;  // original node -> owner (lowest rank among touching parts)
  std::vector<int> new_id;      // original node -> new global node id
  std::vector<int> offsets;     // nparts + 1; rank r owns [offsets[r], offsets[r + 1])
  // Per group, per boundary edge: rank owning the unique adjacent triangle, so each Neumann
  // edge is integrated exactly once and its outward normal can be computed locally.
  std::map<std::string, std::vector<int>> edge_part;
  long long edge_cut = 0;    // METIS objective (dual-graph edge cut); 0 if nparts == 1
  int interface_nodes = 0;   // nodes touched by more than one part
};

enum class Partitioner { Metis, ParMetis };
std::string partitioner_name(Partitioner p);  // "metis" | "parmetis"

// METIS_PartMeshDual (ncommon = 2, fixed seed) + build_partition(); nparts == 1 skips METIS.
// Throws if the mesh does not fit METIS' idx_t, if a node is not used by any triangle, or if
// a boundary edge is not adjacent to exactly one triangle.
GlobalPartition partition_mesh(const Mesh& mesh, int nparts, int seed = kMetisSeed);

// Ownership (lowest touching rank), contiguous global numbering and boundary-edge assignment
// for a given element partition.
GlobalPartition build_partition(const Mesh& mesh, std::vector<int> elem_part, int nparts,
                                long long edge_cut);

// ParMETIS_V3_PartMeshKway (ncommon = 2, fixed seed, uniform tpwgts, ubvec 1.05) on a block
// distribution of the elements scattered from rank 0; the partition vector is gathered back
// to rank 0 (the only rank that holds `mesh`; returned there, empty elsewhere). Collective.
// Throws on every rank if ParMETIS was not compiled in.
std::vector<int> parmetis_element_partition(const Mesh* mesh, MPI_Comm comm, int seed,
                                            long long* edge_cut);

// One rank's part of the distributed mesh.
//
// Local node layout: owned nodes first, [0, num_owned), in global order (global id
// offsets[rank] + i), then ELEMENT GHOSTS: nodes of local triangles owned by other ranks,
// sorted by global id.
//
// The element-ghost set is NOT the SpMV halo of Milestone 4b. The halo comes from the column
// pattern of the owned rows after assembly: an owned row couples to every node of every
// triangle around that node, and some of those triangles live on other ranks, so the halo
// contains nodes that are not in any local triangle (and vice versa: an element ghost need
// not appear in any owned row).
struct DistributedMesh {
  int rank = 0;
  int nranks = 1;
  int global_num_nodes = 0;
  int global_num_triangles = 0;
  std::vector<int> offsets;  // nranks + 1, identical on all ranks
  int num_owned = 0;

  Mesh local;  // xy, node_tags, triangles, boundary_edges (all groups) in local node indices
  std::vector<int> local_to_global;
  std::unordered_map<int, int> global_to_local;
  std::vector<int> node_owner;   // owner rank of each local node
  std::vector<int> element_ids;  // original (file order) index of each local triangle

  int num_local_nodes() const { return static_cast<int>(local_to_global.size()); }
  int num_element_ghosts() const { return num_local_nodes() - num_owned; }
  // Local index of a global node id, or -1 if it is not on this rank.
  int to_local(int global_id) const;
};

// Scatters a partitioned mesh from `root` (the only rank that holds `mesh` and `part`;
// other ranks pass nullptr). Collective.
DistributedMesh distribute_mesh(const Mesh* mesh, const GlobalPartition* part, MPI_Comm comm,
                                int root = 0);

// Global partition statistics, valid on rank 0 only.
struct PartitionSummary {
  std::string mesh_path;
  long long edge_cut = 0;
  int interface_nodes = 0;
  std::string partitioner = "metis";
};

// This rank's wall time in each phase (each phase starts after an MPI_Barrier).
struct PartitionTimings {
  double read = 0.0, partition = 0.0, distribute = 0.0;
};

// Rank 0 reads the mesh; it is partitioned (METIS on rank 0, or ParMETIS on all ranks) and
// distributed. Ownership, numbering and packaging run on rank 0 (v1). Errors are raised on
// every rank. Collective.
DistributedMesh read_partition_distribute(const std::string& mesh_path, MPI_Comm comm,
                                          PartitionSummary* summary = nullptr,
                                          Partitioner partitioner = Partitioner::Metis,
                                          int seed = kMetisSeed,
                                          PartitionTimings* timings = nullptr);

// Per-rank and global partition report, printed by rank 0. Collective.
void print_partition_report(const DistributedMesh& dm, const PartitionSummary& summary,
                            MPI_Comm comm, std::ostream& out);

// Each rank writes <dir>/<base>_<rank>.vtu (mesh, cell partition_id, point owner_rank);
// rank 0 writes <dir>/<base>.pvtu listing all pieces. Creates `dir`. Collective.
void write_partition_vtu(const DistributedMesh& dm, const std::string& dir,
                         const std::string& base, MPI_Comm comm);

}  // namespace cstfem
