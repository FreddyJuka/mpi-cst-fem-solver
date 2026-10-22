#include "cstfem/vtu.hpp"

#include <fstream>
#include <stdexcept>

namespace cstfem {

namespace {

const char* byte_order() {
  const std::uint16_t one = 1;
  return *reinterpret_cast<const unsigned char*>(&one) == 1 ? "LittleEndian" : "BigEndian";
}

// Header (UInt64 byte count) and payload are base64-encoded separately, as VTK does.
std::string encode_block(const void* data, std::size_t size) {
  const std::uint64_t nbytes = size;
  return base64_encode(&nbytes, sizeof nbytes) + base64_encode(data, size);
}

void write_data_array(std::ostream& out, const char* type, const std::string& name,
                      int components, const void* data, std::size_t size) {
  out << "        <DataArray type=\"" << type << "\"";
  if (!name.empty()) out << " Name=\"" << name << "\"";
  out << " NumberOfComponents=\"" << components << "\" format=\"binary\">\n          "
      << encode_block(data, size) << "\n        </DataArray>\n";
}

void write_arrays(std::ostream& out, const std::vector<VtuArray>& arrays, std::size_t count,
                  const char* what) {
  for (const auto& a : arrays) {
    const std::size_t expected = count * static_cast<std::size_t>(a.components);
    const std::size_t actual = a.i32.empty() ? a.f64.size() : a.i32.size();
    if (actual != expected)
      throw std::runtime_error(std::string("vtu: ") + what + " array '" + a.name + "' has " +
                               std::to_string(actual) + " values, expected " +
                               std::to_string(expected));
    if (a.i32.empty())
      write_data_array(out, "Float64", a.name, a.components, a.f64.data(),
                       a.f64.size() * sizeof(double));
    else
      write_data_array(out, "Int32", a.name, a.components, a.i32.data(),
                       a.i32.size() * sizeof(std::int32_t));
  }
}

// Vectors="..." attribute: first 3-component array, so ParaView can warp by it.
std::string vectors_attr(const std::vector<VtuArrayInfo>& infos) {
  for (const auto& i : infos)
    if (i.components == 3) return " Vectors=\"" + i.name + "\"";
  return {};
}

std::ofstream open(const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error(path + ": cannot open for writing");
  return out;
}

}  // namespace

std::string base64_encode(const void* data, std::size_t size) {
  static const char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto* p = static_cast<const unsigned char*>(data);
  std::string out;
  out.reserve(4 * ((size + 2) / 3));
  std::size_t i = 0;
  for (; i + 2 < size; i += 3) {
    const unsigned v = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2];
    out += table[(v >> 18) & 63];
    out += table[(v >> 12) & 63];
    out += table[(v >> 6) & 63];
    out += table[v & 63];
  }
  if (i + 1 == size) {
    const unsigned v = p[i] << 16;
    out += table[(v >> 18) & 63];
    out += table[(v >> 12) & 63];
    out += "==";
  } else if (i + 2 == size) {
    const unsigned v = (p[i] << 16) | (p[i + 1] << 8);
    out += table[(v >> 18) & 63];
    out += table[(v >> 12) & 63];
    out += table[(v >> 6) & 63];
    out += '=';
  }
  return out;
}

std::vector<VtuArrayInfo> array_infos(const std::vector<VtuArray>& arrays) {
  std::vector<VtuArrayInfo> infos;
  for (const auto& a : arrays) infos.push_back({a.name, a.components, !a.i32.empty()});
  return infos;
}

void write_vtu_piece(const std::string& path, const std::vector<double>& xy,
                     const std::vector<int>& triangles, const std::vector<VtuArray>& point_data,
                     const std::vector<VtuArray>& cell_data) {
  const std::size_t np = xy.size() / 2, nc = triangles.size() / 3;
  std::vector<double> points(3 * np);
  for (std::size_t i = 0; i < np; ++i) {
    points[3 * i] = xy[2 * i];
    points[3 * i + 1] = xy[2 * i + 1];
    points[3 * i + 2] = 0.0;
  }
  std::vector<std::int64_t> conn(triangles.begin(), triangles.end()), offsets(nc);
  for (std::size_t c = 0; c < nc; ++c) offsets[c] = static_cast<std::int64_t>(3 * (c + 1));
  const std::vector<std::uint8_t> types(nc, 5);  // VTK_TRIANGLE

  auto out = open(path);
  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"" << byte_order()
      << "\" header_type=\"UInt64\">\n"
      << "  <UnstructuredGrid>\n"
      << "    <Piece NumberOfPoints=\"" << np << "\" NumberOfCells=\"" << nc << "\">\n";
  out << "      <PointData" << vectors_attr(array_infos(point_data)) << ">\n";
  write_arrays(out, point_data, np, "point");
  out << "      </PointData>\n      <CellData>\n";
  write_arrays(out, cell_data, nc, "cell");
  out << "      </CellData>\n      <Points>\n";
  write_data_array(out, "Float64", "Points", 3, points.data(), points.size() * sizeof(double));
  out << "      </Points>\n      <Cells>\n";
  write_data_array(out, "Int64", "connectivity", 1, conn.data(), conn.size() * sizeof(std::int64_t));
  write_data_array(out, "Int64", "offsets", 1, offsets.data(),
                   offsets.size() * sizeof(std::int64_t));
  write_data_array(out, "UInt8", "types", 1, types.data(), types.size());
  out << "      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
  if (!out) throw std::runtime_error(path + ": write failed");
}

void write_pvtu(const std::string& path, const std::vector<std::string>& piece_sources,
                const std::vector<VtuArrayInfo>& point_data,
                const std::vector<VtuArrayInfo>& cell_data) {
  auto pdata = [](std::ostream& out, const VtuArrayInfo& i) {
    out << "      <PDataArray type=\"" << (i.is_int ? "Int32" : "Float64") << "\" Name=\""
        << i.name << "\" NumberOfComponents=\"" << i.components << "\"/>\n";
  };
  auto out = open(path);
  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"" << byte_order()
      << "\" header_type=\"UInt64\">\n"
      << "  <PUnstructuredGrid GhostLevel=\"0\">\n"
      << "    <PPointData" << vectors_attr(point_data) << ">\n";
  for (const auto& i : point_data) pdata(out, i);
  out << "    </PPointData>\n    <PCellData>\n";
  for (const auto& i : cell_data) pdata(out, i);
  out << "    </PCellData>\n    <PPoints>\n"
      << "      <PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\"/>\n"
      << "    </PPoints>\n";
  for (const auto& s : piece_sources) out << "    <Piece Source=\"" << s << "\"/>\n";
  out << "  </PUnstructuredGrid>\n</VTKFile>\n";
  if (!out) throw std::runtime_error(path + ": write failed");
}

}  // namespace cstfem
