#pragma once

#include <array>
#include <cstddef>
#include <istream>
#include <map>
#include <string>
#include <vector>

namespace cstfem {

struct PhysicalGroup {
  int dim = 0;
  int tag = 0;
  std::string name;
};

// Serial 2D triangle mesh. Node and element indices are 0-based.
struct Mesh {
  std::vector<double> xy;               // node coordinates, 2 per node (x0, y0, x1, y1, ...)
  std::vector<std::size_t> node_tags;   // original Gmsh node tag of each node
  std::vector<int> triangles;           // connectivity, 3 per element, counter-clockwise
  std::map<std::string, std::vector<std::array<int, 2>>> boundary_edges;  // by physical name
  std::vector<PhysicalGroup> physical_groups;  // from $PhysicalNames
  int flipped_triangles = 0;            // triangles reoriented to counter-clockwise on read

  std::size_t num_nodes() const { return node_tags.size(); }
  std::size_t num_triangles() const { return triangles.size() / 3; }
  bool has_physical_group(const std::string& name) const;
};

// Reads a Gmsh .msh 4.1 ASCII file. Throws std::runtime_error with "file:line: message".
Mesh read_gmsh(const std::string& path);

// Same as above, reading from a stream; `source_name` is used in error messages.
Mesh read_gmsh(std::istream& in, const std::string& source_name);

}  // namespace cstfem
