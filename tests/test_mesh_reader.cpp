// Unit tests for the Gmsh .msh 4.1 ASCII reader.

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "cstfem/element.hpp"
#include "cstfem/mesh.hpp"
#include "test_assert.hpp"

using namespace cstfem;

namespace {

// Two triangles on the unit square with non-contiguous node tags (7, 10, 20, 35) and
// element tags, an unknown section, a point element, and one clockwise triangle (tag 4).
const char* const kSmallMesh = R"($MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
3
0 3 "corner"
1 1 "bottom"
2 2 "domain"
$EndPhysicalNames
$Entities
1 1 1 0
1 0 0 0 1 3
5 0 0 0 1 0 0 1 1 2 1 -1
9 0 0 0 1 1 0 1 2 1 5
$EndEntities
$Comments
this section is unknown and must be skipped
$EndComments
$Nodes
3 4 7 35
0 1 0 1
10
0 0 0
1 5 0 1
20
1 0 0
2 9 0 2
35
7
1 1 0
0 1 0
$EndNodes
$Elements
3 4 1 40
0 1 15 1
1 10
1 5 1 1
2 10 20
2 9 2 2
30 10 20 35
40 10 7 35
$EndElements
)";

std::string write_temp(const std::string& name, const std::string& content) {
  const std::string path = std::string(CSTFEM_TEST_TMP_DIR) + "/" + name;
  std::ofstream(path) << content;
  return path;
}

std::string replace(std::string s, const std::string& from, const std::string& to) {
  const auto pos = s.find(from);
  if (pos != std::string::npos) s.replace(pos, from.size(), to);
  return s;
}

void test_kirsch_l0() {
  const std::string path = std::string(CSTFEM_TEST_DATA_DIR) + "/kirsch_L0.msh";
  const Mesh m = read_gmsh(path);
  CHECK(m.num_nodes() == 237);
  CHECK(m.num_triangles() == 414);
  CHECK(m.xy.size() == 2 * m.num_nodes());
  CHECK(m.flipped_triangles == 0);
  for (const char* g : {"hole", "left_sym", "bottom_sym", "right", "top", "domain"})
    CHECK(m.has_physical_group(g));
  for (const char* g : {"hole", "left_sym", "bottom_sym", "right", "top"})
    CHECK(m.boundary_edges.count(g) == 1 && !m.boundary_edges.at(g).empty());

  int bad = 0;
  for (const auto& e : m.boundary_edges.at("hole"))
    for (int n : e)
      if (!(std::abs(std::hypot(m.xy[2 * n], m.xy[2 * n + 1]) - 0.1) < 1e-9)) ++bad;
  CHECK(bad == 0);

  // All triangles counter-clockwise after reading.
  int ccw = 0;
  for (std::size_t e = 0; e < m.num_triangles(); ++e) {
    TriCoords xy;
    for (int k = 0; k < 3; ++k) {
      xy[2 * k] = m.xy[2 * m.triangles[3 * e + k]];
      xy[2 * k + 1] = m.xy[2 * m.triangles[3 * e + k] + 1];
    }
    if (cst_signed_area(xy) > 0) ++ccw;
  }
  CHECK(ccw == 414);
}

void test_non_contiguous_tags() {
  const Mesh m = read_gmsh(write_temp("small.msh", kSmallMesh));
  CHECK(m.num_nodes() == 4);
  CHECK(m.num_triangles() == 2);
  CHECK(m.flipped_triangles == 1);
  // Nodes are indexed in file order: tags 10, 20, 35, 7 -> 0, 1, 2, 3.
  CHECK(m.node_tags.size() == 4 && m.node_tags[0] == 10 && m.node_tags[1] == 20 &&
        m.node_tags[2] == 35 && m.node_tags[3] == 7);
  CHECK(m.xy[2 * 3] == 0.0 && m.xy[2 * 3 + 1] == 1.0);  // tag 7 at (0, 1)
  CHECK(m.triangles[0] == 0 && m.triangles[1] == 1 && m.triangles[2] == 2);
  // (10, 7, 35) is clockwise and is flipped to (10, 35, 7) -> (0, 2, 3).
  CHECK(m.triangles[3] == 0 && m.triangles[4] == 2 && m.triangles[5] == 3);
  CHECK(m.boundary_edges.size() == 1);
  CHECK(m.boundary_edges.count("bottom") == 1 && m.boundary_edges.at("bottom").size() == 1);
  CHECK(m.boundary_edges.at("bottom")[0][0] == 0 && m.boundary_edges.at("bottom")[0][1] == 1);
  CHECK(m.has_physical_group("domain") && m.has_physical_group("corner"));
}

void test_errors() {
  const std::string base = kSmallMesh;
  // Messages carry file name and line number.
  CHECK_THROWS(read_gmsh(write_temp("v22.msh", replace(base, "4.1 0 8", "2.2 0 8"))),
               "v22.msh:2: unsupported .msh version '2.2'");
  CHECK_THROWS(read_gmsh(write_temp("bin.msh", replace(base, "4.1 0 8", "4.1 1 8"))),
               "bin.msh:2: binary");
  CHECK_THROWS(read_gmsh(write_temp("quad.msh", replace(base, "2 9 2 2", "2 9 3 2"))),
               "quad.msh:39: unsupported element type 3");
  CHECK_THROWS(read_gmsh(write_temp("tag.msh", replace(base, "2 10 20", "2 10 21"))),
               "tag.msh:38: unknown node tag 21");
  // Degenerate triangle: move node 35 onto the line through 10 and 20.
  CHECK_THROWS(read_gmsh(write_temp("degen.msh", replace(base, "35\n7\n1 1 0\n", "35\n7\n2 0 0\n"))),
               "degen.msh:40: degenerate triangle");
  CHECK_THROWS(read_gmsh(write_temp("trunc.msh", base.substr(0, base.find("$EndNodes")))),
               "trunc.msh:");
  CHECK_THROWS(read_gmsh(std::string(CSTFEM_TEST_TMP_DIR) + "/does_not_exist.msh"),
               "cannot open file");
}

}  // namespace

int main() {
  RUN_TEST(test_kirsch_l0);
  RUN_TEST(test_non_contiguous_tags);
  RUN_TEST(test_errors);
  return test_summary();
}
