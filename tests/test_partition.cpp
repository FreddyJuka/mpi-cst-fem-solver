// MPI test of partitioning, distribution, ownership and global numbering.
// Run with any number of ranks; rank 0 also reads the mesh to compare against.

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cstfem/element.hpp"
#include "cstfem/partition.hpp"

using namespace cstfem;

namespace {

int g_rank = 0, g_size = 1, g_failures = 0;

#define CHECK(cond)                                                                            \
  do {                                                                                         \
    if (!(cond)) {                                                                             \
      std::fprintf(stderr, "[rank %d] %s:%d: CHECK failed: %s\n", g_rank, __FILE__, __LINE__, \
                   #cond);                                                                     \
      ++g_failures;                                                                            \
    }                                                                                          \
  } while (0)

// Gathers a vector of T (as MPI_type) from all ranks to rank 0.
template <class T>
std::vector<T> gather_all(const std::vector<T>& mine, MPI_Datatype type) {
  int n = static_cast<int>(mine.size());
  std::vector<int> counts(g_size), displs(g_size);
  MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  int total = 0;
  for (int r = 0; r < g_size; ++r) {
    displs[r] = total;
    total += counts[r];
  }
  std::vector<T> all(g_rank == 0 ? total : 0);
  MPI_Gatherv(mine.data(), n, type, all.data(), counts.data(), displs.data(), type, 0,
              MPI_COMM_WORLD);
  return all;
}

void check_ownership_and_numbering(const DistributedMesh& dm, const Mesh* full) {
  // Owned nodes are local [0, num_owned) with global ids offsets[rank] + i.
  CHECK(dm.num_owned == dm.offsets[g_rank + 1] - dm.offsets[g_rank]);
  for (int i = 0; i < dm.num_owned; ++i) {
    CHECK(dm.local_to_global[i] == dm.offsets[g_rank] + i);
    CHECK(dm.node_owner[i] == g_rank);
  }
  // Element ghosts: owned elsewhere, by a LOWER rank (owner = lowest touching rank), and the
  // owner is consistent with the offsets.
  for (int i = dm.num_owned; i < dm.num_local_nodes(); ++i) {
    const int gid = dm.local_to_global[i], o = dm.node_owner[i];
    CHECK(o < g_rank);
    CHECK(gid >= dm.offsets[o] && gid < dm.offsets[o + 1]);
    if (i > dm.num_owned) CHECK(dm.local_to_global[i - 1] < gid);
  }

  // Sum of owned = total; offsets = exclusive scan of owned counts.
  int owned = dm.num_owned, total = 0, scan = 0;
  MPI_Allreduce(&owned, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Exscan(&owned, &scan, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) scan = 0;  // MPI_Exscan leaves rank 0 undefined
  CHECK(total == dm.global_num_nodes);
  CHECK(dm.offsets[g_rank] == scan);
  CHECK(dm.offsets.back() == dm.global_num_nodes);

  // Every original node owned exactly once, with its original coordinates.
  std::vector<std::int64_t> tags;
  std::vector<double> xy;
  for (int i = 0; i < dm.num_owned; ++i) {
    tags.push_back(static_cast<std::int64_t>(dm.local.node_tags[i]));
    xy.push_back(dm.local.xy[2 * i]);
    xy.push_back(dm.local.xy[2 * i + 1]);
  }
  const auto all_tags = gather_all(tags, MPI_INT64_T);
  const auto all_xy = gather_all(xy, MPI_DOUBLE);
  if (g_rank == 0) {
    CHECK(all_tags.size() == full->num_nodes());
    std::vector<std::tuple<std::int64_t, double, double>> got, want;
    for (std::size_t i = 0; i < all_tags.size(); ++i)
      got.emplace_back(all_tags[i], all_xy[2 * i], all_xy[2 * i + 1]);
    for (std::size_t n = 0; n < full->num_nodes(); ++n)
      want.emplace_back(static_cast<std::int64_t>(full->node_tags[n]), full->xy[2 * n],
                        full->xy[2 * n + 1]);
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    CHECK(got == want);
  }
}

void check_elements_and_edges(const DistributedMesh& dm, const Mesh* full) {
  // Each triangle exactly once.
  const auto ids = gather_all(dm.element_ids, MPI_INT);
  int ntri = static_cast<int>(dm.local.num_triangles()), sum = 0;
  MPI_Allreduce(&ntri, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  CHECK(sum == dm.global_num_triangles);
  if (g_rank == 0) {
    std::vector<int> sorted = ids, want(full->num_triangles());
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t e = 0; e < want.size(); ++e) want[e] = static_cast<int>(e);
    CHECK(sorted == want);
  }

  // Positive signed area after redistribution and renumbering.
  for (std::size_t e = 0; e < dm.local.num_triangles(); ++e) {
    TriCoords xy;
    for (int k = 0; k < 3; ++k) {
      const int n = dm.local.triangles[3 * e + k];
      xy[2 * k] = dm.local.xy[2 * n];
      xy[2 * k + 1] = dm.local.xy[2 * n + 1];
    }
    CHECK(cst_signed_area(xy) > 0.0);
  }

  // Each boundary edge exactly once per group (as unordered Gmsh tag pairs); its adjacent
  // triangle is local (the outward normal can be computed on this rank).
  for (const auto& [name, edges] : dm.local.boundary_edges) {
    std::vector<std::int64_t> pairs;
    for (const auto& e : edges) {
      const auto a = static_cast<std::int64_t>(dm.local.node_tags[e[0]]);
      const auto b = static_cast<std::int64_t>(dm.local.node_tags[e[1]]);
      pairs.push_back(std::min(a, b));
      pairs.push_back(std::max(a, b));
      int adjacent = 0;
      for (std::size_t t = 0; t < dm.local.num_triangles(); ++t) {
        const int* n = &dm.local.triangles[3 * t];
        const bool has0 = n[0] == e[0] || n[1] == e[0] || n[2] == e[0];
        const bool has1 = n[0] == e[1] || n[1] == e[1] || n[2] == e[1];
        adjacent += has0 && has1;
      }
      CHECK(adjacent == 1);
    }
    const auto all = gather_all(pairs, MPI_INT64_T);
    if (g_rank == 0) {
      std::vector<std::pair<std::int64_t, std::int64_t>> got, want;
      for (std::size_t i = 0; i < all.size(); i += 2) got.emplace_back(all[i], all[i + 1]);
      for (const auto& e : full->boundary_edges.at(name)) {
        const auto a = static_cast<std::int64_t>(full->node_tags[e[0]]);
        const auto b = static_cast<std::int64_t>(full->node_tags[e[1]]);
        want.emplace_back(std::min(a, b), std::max(a, b));
      }
      std::sort(got.begin(), got.end());
      std::sort(want.begin(), want.end());
      CHECK(got == want);
    }
  }
  if (g_rank == 0) CHECK(dm.local.boundary_edges.size() == full->boundary_edges.size());
}

// Every element ghost asks its owner for coordinates and Gmsh tag (actual exchange).
void check_ghost_consistency(const DistributedMesh& dm) {
  std::vector<std::vector<int>> req(g_size);
  for (int i = dm.num_owned; i < dm.num_local_nodes(); ++i)
    req[dm.node_owner[i]].push_back(dm.local_to_global[i]);
  std::vector<int> scount(g_size), rcount(g_size), sdispl(g_size), rdispl(g_size);
  std::vector<int> send;
  for (int r = 0; r < g_size; ++r) {
    sdispl[r] = static_cast<int>(send.size());
    scount[r] = static_cast<int>(req[r].size());
    send.insert(send.end(), req[r].begin(), req[r].end());
  }
  MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
  int nrecv = 0;
  for (int r = 0; r < g_size; ++r) {
    rdispl[r] = nrecv;
    nrecv += rcount[r];
  }
  std::vector<int> asked(nrecv);
  MPI_Alltoallv(send.data(), scount.data(), sdispl.data(), MPI_INT, asked.data(), rcount.data(),
                rdispl.data(), MPI_INT, MPI_COMM_WORLD);

  // Owners answer from their OWNED nodes.
  std::vector<std::int64_t> tag_ans(nrecv);
  std::vector<double> xy_ans(2 * static_cast<std::size_t>(nrecv));
  for (int k = 0; k < nrecv; ++k) {
    const int l = dm.to_local(asked[k]);
    CHECK(l >= 0 && l < dm.num_owned);
    if (l < 0) continue;
    tag_ans[k] = static_cast<std::int64_t>(dm.local.node_tags[l]);
    xy_ans[2 * k] = dm.local.xy[2 * l];
    xy_ans[2 * k + 1] = dm.local.xy[2 * l + 1];
  }
  std::vector<std::int64_t> tag_got(send.size());
  std::vector<double> xy_got(2 * send.size());
  MPI_Alltoallv(tag_ans.data(), rcount.data(), rdispl.data(), MPI_INT64_T, tag_got.data(),
                scount.data(), sdispl.data(), MPI_INT64_T, MPI_COMM_WORLD);
  std::vector<int> rc2(g_size), rd2(g_size), sc2(g_size), sd2(g_size);
  for (int r = 0; r < g_size; ++r) {
    rc2[r] = 2 * rcount[r];
    rd2[r] = 2 * rdispl[r];
    sc2[r] = 2 * scount[r];
    sd2[r] = 2 * sdispl[r];
  }
  MPI_Alltoallv(xy_ans.data(), rc2.data(), rd2.data(), MPI_DOUBLE, xy_got.data(), sc2.data(),
                sd2.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  int mismatches = 0;
  for (std::size_t k = 0; k < send.size(); ++k) {
    const int l = dm.to_local(send[k]);
    if (static_cast<std::int64_t>(dm.local.node_tags[l]) != tag_got[k] ||
        dm.local.xy[2 * l] != xy_got[2 * k] || dm.local.xy[2 * l + 1] != xy_got[2 * k + 1])
      ++mismatches;
  }
  CHECK(mismatches == 0);
}

void check_reproducible(const std::string& path, const DistributedMesh& first) {
  const DistributedMesh again = read_partition_distribute(path, MPI_COMM_WORLD);
  CHECK(again.offsets == first.offsets);
  CHECK(again.local_to_global == first.local_to_global);
  CHECK(again.element_ids == first.element_ids);
  CHECK(again.local.triangles == first.local.triangles);
  CHECK(again.node_owner == first.node_owner);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);

  for (const char* name : {"kirsch_L0.msh", "kirsch_L1.msh"}) {
    const std::string path = std::string(CSTFEM_TEST_DATA_DIR) + "/" + name;
    Mesh full;
    if (g_rank == 0) full = read_gmsh(path);
    PartitionSummary summary;
    const DistributedMesh dm = read_partition_distribute(path, MPI_COMM_WORLD, &summary);
    const int before = g_failures;
    check_ownership_and_numbering(dm, g_rank == 0 ? &full : nullptr);
    check_elements_and_edges(dm, g_rank == 0 ? &full : nullptr);
    check_ghost_consistency(dm);
    check_reproducible(path, dm);
    int local_fail = g_failures - before, fails = 0;
    MPI_Allreduce(&local_fail, &fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int ghosts = dm.num_element_ghosts(), total_ghosts = 0;
    MPI_Reduce(&ghosts, &total_ghosts, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (g_rank == 0)
      std::printf("[%s] %s on %d rank(s): edge cut %lld, interface nodes %d, element ghosts %d\n",
                  fails == 0 ? "PASS" : "FAIL", name, g_size, summary.edge_cut,
                  summary.interface_nodes, total_ghosts);
  }

  int fails = 0;
  MPI_Allreduce(&g_failures, &fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) std::printf("%d failures\n", fails);
  MPI_Finalize();
  return fails == 0 ? 0 : 1;
}
