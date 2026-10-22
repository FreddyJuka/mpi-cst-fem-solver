#pragma once

#include "cstfem/csr.hpp"
#include "cstfem/element.hpp"
#include "cstfem/mesh.hpp"

namespace cstfem {

// DOF numbering: 2 per node, (u0, v0, u1, v1, ...). component: 0 = x, 1 = y.
inline int dof_index(int node, int component) { return 2 * node + component; }

// Sparsity pattern of the global stiffness matrix from node adjacency through elements:
// one 2x2 block per pair of nodes sharing a triangle (including each node with itself).
// Values are zero-initialized.
CsrMatrix build_stiffness_pattern(const Mesh& mesh);

// Element coordinates of triangle `e`.
TriCoords element_coords(const Mesh& mesh, std::size_t e);

// Zeroes K.vals and adds the element stiffness of every triangle.
// K must have the pattern from build_stiffness_pattern(mesh).
void assemble_stiffness(const Mesh& mesh, const Mat3& D, double thickness, CsrMatrix& K);

}  // namespace cstfem
