// Base64 encoder (RFC 4648 test vectors) and VTU/PVTU structure.

#include <fstream>
#include <sstream>
#include <string>

#include "cstfem/vtu.hpp"
#include "test_assert.hpp"

using namespace cstfem;

namespace {

std::string b64(const std::string& s) { return base64_encode(s.data(), s.size()); }

std::string slurp(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void test_base64_rfc4648() {
  // RFC 4648, section 10.
  CHECK(b64("") == "");
  CHECK(b64("f") == "Zg==");
  CHECK(b64("fo") == "Zm8=");
  CHECK(b64("foo") == "Zm9v");
  CHECK(b64("foob") == "Zm9vYg==");
  CHECK(b64("fooba") == "Zm9vYmE=");
  CHECK(b64("foobar") == "Zm9vYmFy");
  // High bytes (no sign-extension problems).
  const unsigned char hi[3] = {0xff, 0xfe, 0x80};
  CHECK(base64_encode(hi, 3) == "//6A");
}

void test_piece_and_master() {
  // Unit square split into two triangles.
  const std::vector<double> xy = {0, 0, 1, 0, 1, 1, 0, 1};
  const std::vector<int> tri = {0, 1, 2, 0, 2, 3};
  std::vector<VtuArray> pd = {{"displacement", 3, std::vector<double>(12, 0.5), {}}};
  std::vector<VtuArray> cd = {{"sigma_xx", 1, {1.0, 2.0}, {}}, {"partition_id", 1, {}, {0, 0}}};
  const std::string dir = CSTFEM_TEST_TMP_DIR;
  write_vtu_piece(dir + "/piece_0.vtu", xy, tri, pd, cd);
  write_pvtu(dir + "/master.pvtu", {"piece_0.vtu"}, array_infos(pd), array_infos(cd));

  const std::string vtu = slurp(dir + "/piece_0.vtu");
  CHECK(vtu.find("type=\"UnstructuredGrid\"") != std::string::npos);
  CHECK(vtu.find("header_type=\"UInt64\"") != std::string::npos);
  CHECK(vtu.find("NumberOfPoints=\"4\" NumberOfCells=\"2\"") != std::string::npos);
  CHECK(vtu.find("<PointData Vectors=\"displacement\">") != std::string::npos);
  CHECK(vtu.find("type=\"Int32\" Name=\"partition_id\"") != std::string::npos);
  // types = {5, 5}: UInt64 header 2 -> "AgAAAAAAAAA=", payload 05 05 -> "BQU=".
  CHECK(vtu.find("AgAAAAAAAAA=BQU=") != std::string::npos);

  const std::string pvtu = slurp(dir + "/master.pvtu");
  CHECK(pvtu.find("type=\"PUnstructuredGrid\"") != std::string::npos);
  CHECK(pvtu.find("<Piece Source=\"piece_0.vtu\"/>") != std::string::npos);
  CHECK(pvtu.find("Name=\"displacement\" NumberOfComponents=\"3\"") != std::string::npos);
  CHECK(pvtu.find("<PDataArray type=\"Int32\" Name=\"partition_id\"") != std::string::npos);

  // Wrong array length is rejected.
  std::vector<VtuArray> bad = {{"sigma_xx", 1, {1.0}, {}}};
  CHECK_THROWS(write_vtu_piece(dir + "/bad.vtu", xy, tri, pd, bad),
               "cell array 'sigma_xx' has 1 values, expected 2");
}

}  // namespace

int main() {
  RUN_TEST(test_base64_rfc4648);
  RUN_TEST(test_piece_and_master);
  return test_summary();
}
