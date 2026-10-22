#include "cstfem/partition.hpp"

#include <metis.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "cstfem/mpi_util.hpp"
#include "cstfem/vtu.hpp"

namespace cstfem {

namespace {

std::uint64_t edge_key(int a, int b) {
  if (a > b) std::swap(a, b);
  return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint32_t>(b);
}

// Serialization of one rank's package: integers as int64, coordinates as double.
struct Writer {
  std::vector<std::int64_t> ints;
  std::vector<double> reals;
  void put(std::int64_t v) { ints.push_back(v); }
  void put(const std::string& s) {
    put(static_cast<std::int64_t>(s.size()));
    for (char c : s) put(static_cast<unsigned char>(c));
  }
};

struct Reader {
  const std::vector<std::int64_t>& ints;
  const std::vector<double>& reals;
  std::size_t i = 0, r = 0;
  std::int64_t get() { return ints.at(i++); }
  int get_int() { return static_cast<int>(get()); }
  std::string get_string() {
    std::string s(static_cast<std::size_t>(get()), '\0');
    for (char& c : s) c = static_cast<char>(get());
    return s;
  }
  double real() { return reals.at(r++); }
};

int checked_count(std::size_t n, const char* what) {
  if (n > static_cast<std::size_t>(INT_MAX))
    throw std::runtime_error(std::string("distribute_mesh: ") + what + " too large for MPI counts");
  return static_cast<int>(n);
}

// Builds the package of rank r on the root.
Writer pack_rank(const Mesh& mesh, const GlobalPartition& p, int r,
                 const std::vector<int>& elems, std::vector<int>& mark) {
  Writer w;
  w.put(static_cast<std::int64_t>(p.offsets.back()));
  w.put(static_cast<std::int64_t>(mesh.num_triangles()));
  for (int o : p.offsets) w.put(o);
  w.put(static_cast<std::int64_t>(mesh.physical_groups.size()));
  for (const auto& g : mesh.physical_groups) {
    w.put(g.dim);
    w.put(g.tag);
    w.put(g.name);
  }

  // Local nodes: owned (by new id) first, then element ghosts (by new id).
  std::vector<int> nodes;
  for (int e : elems)
    for (int k = 0; k < 3; ++k) {
      const int n = mesh.triangles[3 * e + k];
      if (mark[n] != r) {
        mark[n] = r;
        nodes.push_back(n);
      }
    }
  std::sort(nodes.begin(), nodes.end(), [&](int a, int b) {
    const bool ga = p.node_owner[a] != r, gb = p.node_owner[b] != r;
    return ga != gb ? gb : p.new_id[a] < p.new_id[b];
  });
  const auto owned = std::count_if(nodes.begin(), nodes.end(),
                                   [&](int n) { return p.node_owner[n] == r; });
  w.put(static_cast<std::int64_t>(nodes.size()));
  w.put(static_cast<std::int64_t>(owned));
  for (int n : nodes) {
    w.put(p.new_id[n]);
    w.put(p.node_owner[n]);
    w.put(static_cast<std::int64_t>(mesh.node_tags[n]));
    w.reals.push_back(mesh.xy[2 * n]);
    w.reals.push_back(mesh.xy[2 * n + 1]);
  }

  w.put(static_cast<std::int64_t>(elems.size()));
  for (int e : elems) {
    w.put(e);
    for (int k = 0; k < 3; ++k) w.put(p.new_id[mesh.triangles[3 * e + k]]);
  }

  w.put(static_cast<std::int64_t>(mesh.boundary_edges.size()));
  for (const auto& [name, edges] : mesh.boundary_edges) {
    const auto& owner = p.edge_part.at(name);
    w.put(name);
    w.put(static_cast<std::int64_t>(std::count(owner.begin(), owner.end(), r)));
    for (std::size_t i = 0; i < edges.size(); ++i)
      if (owner[i] == r) {
        w.put(p.new_id[edges[i][0]]);
        w.put(p.new_id[edges[i][1]]);
      }
  }
  return w;
}

DistributedMesh unpack(const std::vector<std::int64_t>& ints, const std::vector<double>& reals,
                       int rank, int nranks) {
  Reader rd{ints, reals};
  DistributedMesh dm;
  dm.rank = rank;
  dm.nranks = nranks;
  dm.global_num_nodes = rd.get_int();
  dm.global_num_triangles = rd.get_int();
  dm.offsets.resize(nranks + 1);
  for (int& o : dm.offsets) o = rd.get_int();
  const int nphys = rd.get_int();
  for (int i = 0; i < nphys; ++i) {
    PhysicalGroup g;
    g.dim = rd.get_int();
    g.tag = rd.get_int();
    g.name = rd.get_string();
    dm.local.physical_groups.push_back(g);
  }

  const int nloc = rd.get_int();
  dm.num_owned = rd.get_int();
  dm.local_to_global.resize(nloc);
  dm.node_owner.resize(nloc);
  dm.local.node_tags.resize(nloc);
  dm.local.xy.resize(2 * static_cast<std::size_t>(nloc));
  dm.global_to_local.reserve(nloc);
  for (int i = 0; i < nloc; ++i) {
    dm.local_to_global[i] = rd.get_int();
    dm.node_owner[i] = rd.get_int();
    dm.local.node_tags[i] = static_cast<std::size_t>(rd.get());
    dm.local.xy[2 * i] = rd.real();
    dm.local.xy[2 * i + 1] = rd.real();
    dm.global_to_local.emplace(dm.local_to_global[i], i);
  }

  const int ntri = rd.get_int();
  dm.element_ids.resize(ntri);
  dm.local.triangles.resize(3 * static_cast<std::size_t>(ntri));
  for (int e = 0; e < ntri; ++e) {
    dm.element_ids[e] = rd.get_int();
    for (int k = 0; k < 3; ++k) dm.local.triangles[3 * e + k] = dm.global_to_local.at(rd.get_int());
  }

  const int ngroups = rd.get_int();
  for (int g = 0; g < ngroups; ++g) {
    auto& edges = dm.local.boundary_edges[rd.get_string()];
    edges.resize(static_cast<std::size_t>(rd.get()));
    for (auto& e : edges) {
      e[0] = dm.global_to_local.at(rd.get_int());
      e[1] = dm.global_to_local.at(rd.get_int());
    }
  }
  if (rd.i != ints.size() || rd.r != reals.size())
    throw std::logic_error("distribute_mesh: package size mismatch");
  return dm;
}

}  // namespace

int DistributedMesh::to_local(int global_id) const {
  const auto it = global_to_local.find(global_id);
  return it == global_to_local.end() ? -1 : it->second;
}

GlobalPartition partition_mesh(const Mesh& mesh, int nparts, int seed) {
  const std::size_t ne = mesh.num_triangles(), nn = mesh.num_nodes();
  const auto idx_max = static_cast<std::size_t>(std::numeric_limits<idx_t>::max());
  if (nparts < 1) throw std::runtime_error("partition_mesh: nparts must be >= 1");
  if (nn > idx_max || ne > idx_max / 3)
    throw std::runtime_error("partition_mesh: mesh with " + std::to_string(nn) + " nodes and " +
                             std::to_string(ne) + " triangles does not fit METIS idx_t (" +
                             std::to_string(8 * sizeof(idx_t)) + "-bit)");

  GlobalPartition p;
  p.nparts = nparts;
  p.elem_part.assign(ne, 0);
  if (nparts > 1) {
    if (static_cast<std::size_t>(nparts) > ne)
      throw std::runtime_error("partition_mesh: more parts (" + std::to_string(nparts) +
                               ") than triangles (" + std::to_string(ne) + ")");
    std::vector<idx_t> eptr(ne + 1), eind(3 * ne), epart(ne), npart(nn);
    for (std::size_t e = 0; e <= ne; ++e) eptr[e] = static_cast<idx_t>(3 * e);
    for (std::size_t i = 0; i < 3 * ne; ++i) eind[i] = static_cast<idx_t>(mesh.triangles[i]);
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_SEED] = seed;
    options[METIS_OPTION_NUMBERING] = 0;
    idx_t ne_ = static_cast<idx_t>(ne), nn_ = static_cast<idx_t>(nn), ncommon = 2,
          np_ = nparts, objval = 0;
    const int rc = METIS_PartMeshDual(&ne_, &nn_, eptr.data(), eind.data(), nullptr, nullptr,
                                      &ncommon, &np_, nullptr, options, &objval, epart.data(),
                                      npart.data());
    if (rc != METIS_OK)
      throw std::runtime_error("METIS_PartMeshDual failed (return code " + std::to_string(rc) +
                               ")");
    for (std::size_t e = 0; e < ne; ++e) p.elem_part[e] = static_cast<int>(epart[e]);
    p.edge_cut = objval;
  }

  // Ownership: lowest rank among the parts touching the node.
  p.node_owner.assign(nn, INT_MAX);
  std::vector<int> first_part(nn, -1);
  std::vector<char> multi(nn, 0);
  for (std::size_t e = 0; e < ne; ++e)
    for (int k = 0; k < 3; ++k) {
      const int n = mesh.triangles[3 * e + k], q = p.elem_part[e];
      p.node_owner[n] = std::min(p.node_owner[n], q);
      if (first_part[n] < 0)
        first_part[n] = q;
      else if (first_part[n] != q)
        multi[n] = 1;
    }
  for (std::size_t n = 0; n < nn; ++n) {
    if (p.node_owner[n] == INT_MAX)
      throw std::runtime_error("partition_mesh: node tag " + std::to_string(mesh.node_tags[n]) +
                               " is not used by any triangle");
    p.interface_nodes += multi[n];
  }

  // Contiguous global numbering per owner, keeping file order within a rank.
  p.offsets.assign(nparts + 1, 0);
  for (std::size_t n = 0; n < nn; ++n) ++p.offsets[p.node_owner[n] + 1];
  for (int r = 0; r < nparts; ++r) p.offsets[r + 1] += p.offsets[r];
  std::vector<int> cursor(p.offsets.begin(), p.offsets.end() - 1);
  p.new_id.resize(nn);
  for (std::size_t n = 0; n < nn; ++n) p.new_id[n] = cursor[p.node_owner[n]]++;

  // Boundary edges go to the rank of their unique adjacent triangle.
  for (const auto& [name, edges] : mesh.boundary_edges) {
    std::unordered_map<std::uint64_t, int> tri_of;  // edge -> adjacent triangle (-2: several)
    for (const auto& e : edges) tri_of.emplace(edge_key(e[0], e[1]), -1);
    for (std::size_t t = 0; t < ne; ++t)
      for (int k = 0; k < 3; ++k) {
        const auto it =
            tri_of.find(edge_key(mesh.triangles[3 * t + k], mesh.triangles[3 * t + (k + 1) % 3]));
        if (it != tri_of.end()) it->second = it->second == -1 ? static_cast<int>(t) : -2;
      }
    auto& owner = p.edge_part[name];
    for (const auto& e : edges) {
      const int t = tri_of.at(edge_key(e[0], e[1]));
      if (t < 0)
        throw std::runtime_error("partition_mesh: edge (" + std::to_string(mesh.node_tags[e[0]]) +
                                 ", " + std::to_string(mesh.node_tags[e[1]]) + ") of group '" +
                                 name + "' is not adjacent to exactly one triangle");
      owner.push_back(p.elem_part[t]);
    }
  }
  return p;
}

DistributedMesh distribute_mesh(const Mesh* mesh, const GlobalPartition* part, MPI_Comm comm,
                                int root) {
  int rank = 0, nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);

  std::vector<std::int64_t> all_ints;
  std::vector<double> all_reals;
  std::vector<int> int_counts(nranks), real_counts(nranks), int_displs(nranks),
      real_displs(nranks);
  bool ok = true;
  std::string error;
  if (rank == root) {
    try {
      if (!mesh || !part) throw std::logic_error("distribute_mesh: root needs mesh and partition");
      if (part->nparts != nranks)
        throw std::runtime_error("distribute_mesh: partition has " + std::to_string(part->nparts) +
                                 " parts for " + std::to_string(nranks) + " ranks");
      std::vector<std::vector<int>> elems(nranks);
      for (std::size_t e = 0; e < mesh->num_triangles(); ++e)
        elems[part->elem_part[e]].push_back(static_cast<int>(e));
      std::vector<int> mark(mesh->num_nodes(), -1);
      for (int r = 0; r < nranks; ++r) {
        Writer w = pack_rank(*mesh, *part, r, elems[r], mark);
        int_displs[r] = checked_count(all_ints.size(), "package");
        real_displs[r] = checked_count(all_reals.size(), "package");
        int_counts[r] = checked_count(w.ints.size(), "package");
        real_counts[r] = checked_count(w.reals.size(), "package");
        all_ints.insert(all_ints.end(), w.ints.begin(), w.ints.end());
        all_reals.insert(all_reals.end(), w.reals.begin(), w.reals.end());
        checked_count(all_ints.size(), "total package");
      }
    } catch (const std::exception& e) {
      ok = false;
      error = e.what();
    }
  }
  check_collective(ok, error, comm);

  int counts[2] = {0, 0};
  std::vector<int> pairs;
  if (rank == root)
    for (int r = 0; r < nranks; ++r) pairs.insert(pairs.end(), {int_counts[r], real_counts[r]});
  MPI_Scatter(pairs.data(), 2, MPI_INT, counts, 2, MPI_INT, root, comm);
  std::vector<std::int64_t> ints(counts[0]);
  std::vector<double> reals(counts[1]);
  MPI_Scatterv(all_ints.data(), int_counts.data(), int_displs.data(), MPI_INT64_T, ints.data(),
               counts[0], MPI_INT64_T, root, comm);
  MPI_Scatterv(all_reals.data(), real_counts.data(), real_displs.data(), MPI_DOUBLE,
               reals.data(), counts[1], MPI_DOUBLE, root, comm);
  return unpack(ints, reals, rank, nranks);
}

DistributedMesh read_partition_distribute(const std::string& mesh_path, MPI_Comm comm,
                                          PartitionSummary* summary, int seed) {
  int rank = 0, nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);
  Mesh mesh;
  GlobalPartition part;
  bool ok = true;
  std::string error;
  if (rank == 0) {
    try {
      mesh = read_gmsh(mesh_path);
      part = partition_mesh(mesh, nranks, seed);
      if (summary) *summary = {mesh_path, part.edge_cut, part.interface_nodes};
    } catch (const std::exception& e) {
      ok = false;
      error = e.what();
    }
  }
  check_collective(ok, error, comm);
  return distribute_mesh(rank == 0 ? &mesh : nullptr, rank == 0 ? &part : nullptr, comm, 0);
}

void print_partition_report(const DistributedMesh& dm, const PartitionSummary& summary,
                            MPI_Comm comm, std::ostream& out) {
  std::vector<std::string> groups;
  for (const auto& kv : dm.local.boundary_edges) groups.push_back(kv.first);
  std::vector<long long> mine = {static_cast<long long>(dm.local.num_triangles()), dm.num_owned,
                                 dm.num_element_ghosts()};
  for (const auto& g : groups) mine.push_back(static_cast<long long>(dm.local.boundary_edges.at(g).size()));
  const int width = static_cast<int>(mine.size());
  std::vector<long long> all(dm.rank == 0 ? width * dm.nranks : 0);
  MPI_Gather(mine.data(), width, MPI_LONG_LONG, all.data(), width, MPI_LONG_LONG, 0, comm);
  if (dm.rank != 0) return;

  out << "Partition report: " << summary.mesh_path << ", " << dm.nranks
      << " rank(s), METIS_PartMeshDual (ncommon = 2, seed = " << kMetisSeed << ")"
      << (dm.nranks == 1 ? " skipped for 1 rank" : "") << "\n";
  out << std::setw(5) << "rank" << std::setw(11) << "triangles" << std::setw(8) << "owned"
      << std::setw(13) << "elem-ghosts";
  for (const auto& g : groups) out << std::setw(std::max<int>(11, static_cast<int>(g.size()) + 2)) << g;
  out << "\n";
  long long tmax = 0, ttot = 0;
  for (int r = 0; r < dm.nranks; ++r) {
    const long long* row = &all[static_cast<std::size_t>(r) * width];
    out << std::setw(5) << r << std::setw(11) << row[0] << std::setw(8) << row[1] << std::setw(13)
        << row[2];
    for (std::size_t g = 0; g < groups.size(); ++g)
      out << std::setw(std::max<int>(11, static_cast<int>(groups[g].size()) + 2)) << row[3 + g];
    out << "\n";
    tmax = std::max(tmax, row[0]);
    ttot += row[0];
  }
  const double mean = static_cast<double>(ttot) / dm.nranks;
  out << "global: " << dm.global_num_nodes << " nodes, " << dm.global_num_triangles
      << " triangles, METIS edge cut " << summary.edge_cut << ", interface nodes "
      << summary.interface_nodes << ", load imbalance (max/mean triangles) " << std::fixed
      << std::setprecision(4) << (mean > 0 ? tmax / mean : 0.0) << std::defaultfloat << "\n";
}

void write_partition_vtu(const DistributedMesh& dm, const std::string& dir,
                         const std::string& base, MPI_Comm comm) {
  bool ok = true;
  std::string error;
  if (dm.rank == 0) {
    try {
      std::filesystem::create_directories(dir);
      // Remove pieces left over from an earlier run with more ranks.
      for (int r = dm.nranks;; ++r) {
        if (!std::filesystem::remove(dir + "/" + base + "_" + std::to_string(r) + ".vtu")) break;
      }
    } catch (const std::exception& e) {
      ok = false;
      error = e.what();
    }
  }
  check_collective(ok, error, comm);

  const std::vector<VtuArray> point_data = {{"owner_rank", 1, {}, {dm.node_owner.begin(), dm.node_owner.end()}}};
  const std::vector<VtuArray> cell_data = {
      {"partition_id", 1, {}, std::vector<std::int32_t>(dm.local.num_triangles(), dm.rank)}};
  try {
    write_vtu_piece(dir + "/" + base + "_" + std::to_string(dm.rank) + ".vtu", dm.local.xy,
                    dm.local.triangles, point_data, cell_data);
  } catch (const std::exception& e) {
    ok = false;
    error = e.what();
  }
  check_collective(ok, error, comm);

  if (dm.rank == 0) {
    std::vector<std::string> pieces;
    for (int r = 0; r < dm.nranks; ++r) pieces.push_back(base + "_" + std::to_string(r) + ".vtu");
    try {
      write_pvtu(dir + "/" + base + ".pvtu", pieces, array_infos(point_data),
                 array_infos(cell_data));
    } catch (const std::exception& e) {
      ok = false;
      error = e.what();
    }
  }
  check_collective(ok, error, comm);
}

}  // namespace cstfem
