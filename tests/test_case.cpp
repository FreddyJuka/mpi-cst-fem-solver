// Unit tests for the key = value case-file parser.

#include <sstream>
#include <string>

#include "cstfem/case.hpp"
#include "test_assert.hpp"

using namespace cstfem;

namespace {

CaseFile parse(const std::string& text) {
  std::istringstream in(text);
  return CaseFile::parse(in, "test.case");
}

void test_valid_file() {
  const CaseFile c = parse(R"(# full-line comment
mesh            = meshes/kirsch_L3.msh
plane           = plane_stress

E               = 210e9     # trailing comment
nu              = 0.3
max_iter        = 20000
bc.dirichlet    = left_sym:ux=0 ; bottom_sym:uy=0
   kirsch.a=0.1
)");
  CHECK(c.get_string("mesh") == "meshes/kirsch_L3.msh");
  CHECK(c.get_string("plane") == "plane_stress");
  CHECK(c.get_double("E") == 210e9);
  CHECK(c.get_double("nu") == 0.3);
  CHECK(c.get_int("max_iter") == 20000);
  CHECK(c.get_string("bc.dirichlet") == "left_sym:ux=0 ; bottom_sym:uy=0");
  CHECK(c.get_double("kirsch.a") == 0.1);
  // Optional getters fall back only when the key is absent.
  CHECK(c.get_double("rtol", 1e-10) == 1e-10);
  CHECK(c.get_int("max_iter", 5) == 20000);
  CHECK(c.get_string("solver", "cg_jacobi") == "cg_jacobi");
  CHECK(!c.has("solver") && c.has("E"));
}

void test_unknown_keys() {
  const CaseFile c = parse("mesh = a.msh\nmseh = typo\nE = 1\n");
  const auto unknown = c.unknown_keys({"mesh", "E"});
  CHECK(unknown.size() == 1 && unknown[0] == "mseh");
  std::ostringstream os;
  CHECK(c.warn_unknown_keys({"mesh", "E"}, os) == 1);
  CHECK(os.str().find("test.case:2: unknown key 'mseh'") != std::string::npos);
}

void test_errors() {
  CHECK_THROWS(parse("E = 1\nnu = 0.3\nE = 2\n"),
               "test.case:3: duplicate key 'E' (first defined on line 1)");
  CHECK_THROWS(parse("E = 1\n").get_double("nu"), "missing required key 'nu'");
  CHECK_THROWS(parse("E = 1\n").get_string("mesh"), "missing required key 'mesh'");
  CHECK_THROWS(parse("\nE = 21O e9\n").get_double("E"), "test.case:2: key 'E': '21O e9' is not a number");
  CHECK_THROWS(parse("E =\n").get_double("E"), "is not a number");
  CHECK_THROWS(parse("n = 1.5\n").get_int("n"), "test.case:1: key 'n': '1.5' is not an integer");
  CHECK_THROWS(parse("n = 99999999999\n").get_int("n"), "out of range");
  CHECK_THROWS(parse("just a line\n"), "test.case:1: expected 'key = value'");
  CHECK_THROWS(parse(" = 3\n"), "test.case:1: empty key");
  CHECK_THROWS(CaseFile::load("/nonexistent/x.case"), "cannot open file");
}

}  // namespace

int main() {
  RUN_TEST(test_valid_file);
  RUN_TEST(test_unknown_keys);
  RUN_TEST(test_errors);
  return test_summary();
}
