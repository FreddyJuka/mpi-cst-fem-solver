#include "cstfem/assembly.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cstfem {

CsrMatrix build_stiffness_pattern(const Mesh& mesh) {
  const int nn = static_cast<int>(mesh.num_nodes());
  std::vector<std::vector<int>> adj(nn);
  for (int n = 0; n < nn; ++n) adj[n].push_back(n);
  for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
    const int* t = &mesh.triangles[3 * e];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        if (a != b) adj[t[a]].push_back(t[b]);
  }

  CsrMatrix K;
  K.num_rows = K.num_cols = 2 * nn;
  K.row_ptr.assign(K.num_rows + 1, 0);
  for (int n = 0; n < nn; ++n) {
    auto& a = adj[n];
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
    const int len = 2 * static_cast<int>(a.size());
    K.row_ptr[2 * n + 1] = K.row_ptr[2 * n] + len;
    K.row_ptr[2 * n + 2] = K.row_ptr[2 * n + 1] + len;
  }
  K.cols.reserve(K.row_ptr.back());
  for (int n = 0; n < nn; ++n) {
    for (int c = 0; c < 2; ++c) {
      for (int m : adj[n]) {
        K.cols.push_back(dof_index(m, 0));
        K.cols.push_back(dof_index(m, 1));
      }
    }
  }
  K.vals.assign(K.cols.size(), 0.0);
  return K;
}

TriCoords element_coords(const Mesh& mesh, std::size_t e) {
  TriCoords xy;
  for (int k = 0; k < 3; ++k) {
    const int n = mesh.triangles[3 * e + k];
    xy[2 * k] = mesh.xy[2 * n];
    xy[2 * k + 1] = mesh.xy[2 * n + 1];
  }
  return xy;
}

void assemble_stiffness(const Mesh& mesh, const Mat3& D, double thickness, CsrMatrix& K) {
  std::fill(K.vals.begin(), K.vals.end(), 0.0);
  for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
    const Mat6 Ke = cst_stiffness(element_coords(mesh, e), D, thickness);
    int dofs[6];
    for (int k = 0; k < 3; ++k) {
      dofs[2 * k] = dof_index(mesh.triangles[3 * e + k], 0);
      dofs[2 * k + 1] = dof_index(mesh.triangles[3 * e + k], 1);
    }
    for (int i = 0; i < 6; ++i) {
      const int row = dofs[i] - K.first_row;
      for (int j = 0; j < 6; ++j) {
        const int pos = K.find(row, dofs[j]);
        if (pos < 0)
          throw std::logic_error("assemble_stiffness: entry (" + std::to_string(dofs[i]) + ", " +
                                 std::to_string(dofs[j]) + ") missing from the pattern");
        K.vals[pos] += Ke[i * 6 + j];
      }
    }
  }
}

}  // namespace cstfem
