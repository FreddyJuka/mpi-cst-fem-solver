#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cstfem {

// RFC 4648 base64 (standard alphabet, '=' padding).
std::string base64_encode(const void* data, std::size_t size);

// A named point or cell array: Int32 if `i32` is non-empty, else Int64 if `i64` is
// non-empty, else Float64.
struct VtuArray {
  std::string name;
  int components = 1;
  std::vector<double> f64;
  std::vector<std::int32_t> i32;
  std::vector<std::int64_t> i64;
};

// Name, type and component count of an array (what a .pvtu master needs).
struct VtuArrayInfo {
  std::string name;
  int components = 1;
  std::string type = "Float64";  // Float64 | Int32 | Int64
};
std::vector<VtuArrayInfo> array_infos(const std::vector<VtuArray>& arrays);

// Writes one XML UnstructuredGrid piece of linear triangles (VTK type 5), inline base64
// binary, header_type UInt64, no compression. `xy` has 2 values per point (z = 0).
// Point arrays have size num_points * components, cell arrays num_cells * components.
void write_vtu_piece(const std::string& path, const std::vector<double>& xy,
                     const std::vector<int>& triangles, const std::vector<VtuArray>& point_data,
                     const std::vector<VtuArray>& cell_data);

// Writes the .pvtu master listing `piece_sources` (paths relative to the .pvtu).
void write_pvtu(const std::string& path, const std::vector<std::string>& piece_sources,
                const std::vector<VtuArrayInfo>& point_data,
                const std::vector<VtuArrayInfo>& cell_data);

// Deletes <dir>/<base>_<r>.vtu for r = first, first + 1, ... while such files exist (pieces
// left over from an earlier run with more ranks).
void remove_stale_pieces(const std::string& dir, const std::string& base, int first);

}  // namespace cstfem
