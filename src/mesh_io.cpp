#include "cstfem/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "cstfem/element.hpp"

namespace cstfem {

bool Mesh::has_physical_group(const std::string& name) const {
  return std::any_of(physical_groups.begin(), physical_groups.end(),
                     [&](const PhysicalGroup& g) { return g.name == name; });
}

namespace {

// Reads non-blank lines and keeps track of the line number for error messages.
class LineReader {
public:
  LineReader(std::istream& in, std::string source) : in_(in), source_(std::move(source)) {}

  bool next(std::string& line) {
    while (std::getline(in_, line)) {
      ++line_no_;
      const auto first = line.find_first_not_of(" \t\r");
      if (first == std::string::npos) continue;
      const auto last = line.find_last_not_of(" \t\r");
      line = line.substr(first, last - first + 1);
      return true;
    }
    return false;
  }

  std::string require(const std::string& what) {
    std::string line;
    if (!next(line)) fail("unexpected end of file while reading " + what);
    return line;
  }

  [[noreturn]] void fail(const std::string& msg) const {
    throw std::runtime_error(source_ + ":" + std::to_string(line_no_) + ": " + msg);
  }

private:
  std::istream& in_;
  std::string source_;
  int line_no_ = 0;
};

// Whitespace-separated values of one line, with errors pointing at that line.
class Tokens {
public:
  Tokens(LineReader& r, const std::string& line) : r_(r), ss_(line) {}

  long long integer(const char* what) {
    long long v = 0;
    if (!(ss_ >> v)) r_.fail(std::string("expected integer (") + what + ")");
    return v;
  }

  double real(const char* what) {
    double v = 0.0;
    if (!(ss_ >> v)) r_.fail(std::string("expected number (") + what + ")");
    return v;
  }

  std::string rest() {
    std::string s;
    std::getline(ss_, s);
    const auto first = s.find_first_not_of(" \t");
    return first == std::string::npos ? std::string() : s.substr(first);
  }

private:
  LineReader& r_;
  std::istringstream ss_;
};

int to_int(LineReader& r, long long v, const char* what) {
  if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
    r.fail(std::string(what) + " out of range");
  return static_cast<int>(v);
}

std::size_t to_count(LineReader& r, long long v, const char* what) {
  if (v < 0) r.fail(std::string(what) + " must be non-negative");
  return static_cast<std::size_t>(v);
}

void expect_end(LineReader& r, const std::string& section) {
  const std::string line = r.require("$End" + section);
  if (line != "$End" + section) r.fail("expected $End" + section + ", got '" + line + "'");
}

using EntityKey = std::pair<int, int>;  // (dim, tag)

struct ParseState {
  bool have_format = false;
  bool have_names = false;
  bool have_entities = false;
  bool have_nodes = false;
  bool have_elements = false;
  std::map<EntityKey, std::string> names;              // physical (dim, tag) -> name
  std::map<EntityKey, std::vector<int>> entity_phys;   // entity (dim, tag) -> physical tags
  std::unordered_map<std::size_t, int> node_index;     // node tag -> 0-based index
};

void parse_format(LineReader& r, ParseState& st) {
  const std::string line = r.require("$MeshFormat");
  const std::string version = line.substr(0, line.find_first_of(" \t"));
  if (version != "4.1") r.fail("unsupported .msh version '" + version + "' (need 4.1)");
  Tokens t(r, line);
  t.real("version");
  const long long file_type = t.integer("file-type");
  if (file_type != 0) r.fail("binary .msh files are not supported (need ASCII, file-type 0)");
  t.integer("data-size");
  expect_end(r, "MeshFormat");
  st.have_format = true;
}

void parse_physical_names(LineReader& r, ParseState& st, Mesh& mesh) {
  Tokens head(r, r.require("$PhysicalNames"));
  const std::size_t n = to_count(r, head.integer("numPhysicalNames"), "numPhysicalNames");
  for (std::size_t i = 0; i < n; ++i) {
    Tokens t(r, r.require("physical name"));
    const int dim = to_int(r, t.integer("dimension"), "dimension");
    const int tag = to_int(r, t.integer("physical tag"), "physical tag");
    std::string name = t.rest();
    if (name.size() < 2 || name.front() != '"' || name.back() != '"')
      r.fail("physical name must be quoted");
    name = name.substr(1, name.size() - 2);
    if (!st.names.emplace(EntityKey{dim, tag}, name).second)
      r.fail("duplicate physical group (" + std::to_string(dim) + ", " + std::to_string(tag) +
             ")");
    mesh.physical_groups.push_back({dim, tag, name});
  }
  expect_end(r, "PhysicalNames");
  st.have_names = true;
}

void parse_entities(LineReader& r, ParseState& st) {
  Tokens head(r, r.require("$Entities"));
  std::size_t count[4];
  count[0] = to_count(r, head.integer("numPoints"), "numPoints");
  count[1] = to_count(r, head.integer("numCurves"), "numCurves");
  count[2] = to_count(r, head.integer("numSurfaces"), "numSurfaces");
  count[3] = to_count(r, head.integer("numVolumes"), "numVolumes");
  for (int dim = 0; dim < 4; ++dim) {
    for (std::size_t i = 0; i < count[dim]; ++i) {
      Tokens t(r, r.require("entity"));
      const int tag = to_int(r, t.integer("entity tag"), "entity tag");
      // Points store x y z; higher dimensions store a bounding box (6 values).
      const int ncoords = dim == 0 ? 3 : 6;
      for (int k = 0; k < ncoords; ++k) t.real("entity coordinate");
      const std::size_t nphys = to_count(r, t.integer("numPhysicalTags"), "numPhysicalTags");
      std::vector<int> phys(nphys);
      for (auto& p : phys) p = to_int(r, t.integer("physical tag"), "physical tag");
      // Bounding entities (dim > 0) are not needed.
      st.entity_phys[{dim, tag}] = std::move(phys);
    }
  }
  expect_end(r, "Entities");
  st.have_entities = true;
}

void parse_nodes(LineReader& r, ParseState& st, Mesh& mesh) {
  Tokens head(r, r.require("$Nodes"));
  const std::size_t nblocks = to_count(r, head.integer("numEntityBlocks"), "numEntityBlocks");
  const std::size_t nnodes = to_count(r, head.integer("numNodes"), "numNodes");
  mesh.xy.reserve(2 * nnodes);
  mesh.node_tags.reserve(nnodes);
  st.node_index.reserve(nnodes);

  for (std::size_t b = 0; b < nblocks; ++b) {
    Tokens bh(r, r.require("node block header"));
    bh.integer("entityDim");
    bh.integer("entityTag");
    bh.integer("parametric");
    const std::size_t n = to_count(r, bh.integer("numNodesInBlock"), "numNodesInBlock");
    for (std::size_t i = 0; i < n; ++i) {
      Tokens t(r, r.require("node tag"));
      const long long tag = t.integer("node tag");
      if (tag <= 0) r.fail("node tag must be positive");
      const auto utag = static_cast<std::size_t>(tag);
      const int index = static_cast<int>(mesh.node_tags.size());
      if (!st.node_index.emplace(utag, index).second)
        r.fail("duplicate node tag " + std::to_string(tag));
      mesh.node_tags.push_back(utag);
    }
    for (std::size_t i = 0; i < n; ++i) {
      Tokens t(r, r.require("node coordinates"));
      const double x = t.real("x");
      const double y = t.real("y");
      t.real("z");  // ignored: the mesh is planar; parametric coordinates are ignored too
      mesh.xy.push_back(x);
      mesh.xy.push_back(y);
    }
  }
  if (mesh.node_tags.size() != nnodes)
    r.fail("node count mismatch: header says " + std::to_string(nnodes) + ", blocks contain " +
           std::to_string(mesh.node_tags.size()));
  expect_end(r, "Nodes");
  st.have_nodes = true;
}

void parse_elements(LineReader& r, ParseState& st, Mesh& mesh) {
  if (!st.have_nodes) r.fail("$Elements must come after $Nodes");
  if (!st.have_entities) r.fail("$Elements must come after $Entities");

  // Degenerate-area threshold relative to the bounding-box diagonal.
  double xmin = std::numeric_limits<double>::max(), xmax = std::numeric_limits<double>::lowest();
  double ymin = xmin, ymax = xmax;
  for (std::size_t i = 0; i < mesh.num_nodes(); ++i) {
    xmin = std::min(xmin, mesh.xy[2 * i]);
    xmax = std::max(xmax, mesh.xy[2 * i]);
    ymin = std::min(ymin, mesh.xy[2 * i + 1]);
    ymax = std::max(ymax, mesh.xy[2 * i + 1]);
  }
  const double diag2 =
      mesh.num_nodes() ? (xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) : 0.0;
  const double min_area = 1e-14 * diag2;

  auto node = [&](long long tag) {
    const auto it = st.node_index.find(static_cast<std::size_t>(tag));
    if (tag <= 0 || it == st.node_index.end()) r.fail("unknown node tag " + std::to_string(tag));
    return it->second;
  };

  Tokens head(r, r.require("$Elements"));
  const std::size_t nblocks = to_count(r, head.integer("numEntityBlocks"), "numEntityBlocks");
  const std::size_t nelems = to_count(r, head.integer("numElements"), "numElements");
  std::size_t seen = 0;

  for (std::size_t b = 0; b < nblocks; ++b) {
    Tokens bh(r, r.require("element block header"));
    const int edim = to_int(r, bh.integer("entityDim"), "entityDim");
    const int etag = to_int(r, bh.integer("entityTag"), "entityTag");
    const int type = to_int(r, bh.integer("elementType"), "elementType");
    const std::size_t n = to_count(r, bh.integer("numElementsInBlock"), "numElementsInBlock");
    seen += n;

    int nodes_per_elem = 0;
    switch (type) {
    case 1: nodes_per_elem = 2; break;
    case 2: nodes_per_elem = 3; break;
    case 15: nodes_per_elem = 1; break;
    default:
      r.fail("unsupported element type " + std::to_string(type) +
             " (supported: 1 = 2-node line, 2 = 3-node triangle, 15 = point)");
    }

    // Physical group names of the block's entity (lines only need them).
    std::vector<std::string> group_names;
    if (type == 1) {
      const auto it = st.entity_phys.find({edim, etag});
      if (it != st.entity_phys.end()) {
        for (int p : it->second) {
          const auto nit = st.names.find({edim, p});
          group_names.push_back(nit != st.names.end() ? nit->second : std::to_string(p));
        }
      }
    }

    for (std::size_t i = 0; i < n; ++i) {
      Tokens t(r, r.require("element"));
      t.integer("element tag");
      int conn[3] = {0, 0, 0};
      for (int k = 0; k < nodes_per_elem; ++k) conn[k] = node(t.integer("element node tag"));

      if (type == 2) {
        const std::array<double, 6> xy = {
            mesh.xy[2 * conn[0]], mesh.xy[2 * conn[0] + 1], mesh.xy[2 * conn[1]],
            mesh.xy[2 * conn[1] + 1], mesh.xy[2 * conn[2]], mesh.xy[2 * conn[2] + 1]};
        const double area = cst_signed_area(xy);
        if (area == 0.0 || std::abs(area) < min_area) {
          std::ostringstream msg;
          msg << "degenerate triangle (area " << area << ", threshold " << min_area << ")";
          r.fail(msg.str());
        }
        if (area < 0.0) {
          std::swap(conn[1], conn[2]);
          ++mesh.flipped_triangles;
        }
        mesh.triangles.insert(mesh.triangles.end(), conn, conn + 3);
      } else if (type == 1) {
        for (const auto& name : group_names) mesh.boundary_edges[name].push_back({conn[0], conn[1]});
      }
    }
  }
  if (seen != nelems)
    r.fail("element count mismatch: header says " + std::to_string(nelems) +
           ", blocks contain " + std::to_string(seen));
  expect_end(r, "Elements");
  st.have_elements = true;
}

void skip_section(LineReader& r, const std::string& section) {
  std::string line;
  while (r.next(line)) {
    if (line == "$End" + section) return;
  }
  r.fail("unexpected end of file inside $" + section);
}

}  // namespace

Mesh read_gmsh(std::istream& in, const std::string& source_name) {
  LineReader r(in, source_name);
  ParseState st;
  Mesh mesh;

  std::string line;
  while (r.next(line)) {
    if (line.empty() || line.front() != '$' || line.rfind("$End", 0) == 0)
      r.fail("expected a section header, got '" + line + "'");
    const std::string section = line.substr(1);
    if (!st.have_format && section != "MeshFormat") r.fail("file must start with $MeshFormat");

    if (section == "MeshFormat") {
      if (st.have_format) r.fail("duplicate $MeshFormat section");
      parse_format(r, st);
    } else if (section == "PhysicalNames") {
      parse_physical_names(r, st, mesh);
    } else if (section == "Entities") {
      parse_entities(r, st);
    } else if (section == "Nodes") {
      parse_nodes(r, st, mesh);
    } else if (section == "Elements") {
      parse_elements(r, st, mesh);
    } else {
      skip_section(r, section);
    }
  }

  const std::pair<bool, const char*> required[] = {{st.have_format, "$MeshFormat"},
                                                   {st.have_names, "$PhysicalNames"},
                                                   {st.have_entities, "$Entities"},
                                                   {st.have_nodes, "$Nodes"},
                                                   {st.have_elements, "$Elements"}};
  for (const auto& [present, name] : required) {
    if (!present) r.fail(std::string("missing ") + name + " section");
  }
  return mesh;
}

Mesh read_gmsh(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error(path + ": cannot open file");
  return read_gmsh(in, path);
}

}  // namespace cstfem
