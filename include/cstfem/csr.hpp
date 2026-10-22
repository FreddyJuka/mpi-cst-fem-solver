#pragma once

#include <cstddef>
#include <vector>

namespace cstfem {

// Compressed sparse row matrix with sorted column indices in each row.
//
// Rows and columns live in separate index spaces: rows are local (0 .. num_rows-1), columns
// are indices into a column space of size num_cols. Local row i corresponds to column
// first_row + i. In the serial code first_row = 0 and num_rows = num_cols; in the
// distributed version (Milestone 4) a rank stores only its owned rows with global columns.
struct CsrMatrix {
  int num_rows = 0;
  int num_cols = 0;
  int first_row = 0;
  std::vector<int> row_ptr;  // size num_rows + 1
  std::vector<int> cols;     // size nnz, sorted within each row
  std::vector<double> vals;  // size nnz

  std::size_t nnz() const { return cols.size(); }

  // Position of entry (row, col) in cols/vals, or -1 if it is not in the pattern.
  int find(int row, int col) const;

  // Diagonal entry of local row `row` (0 if not in the pattern).
  double diagonal(int row) const;

  // y = A x; x has num_cols entries, y is resized to num_rows.
  void multiply(const std::vector<double>& x, std::vector<double>& y) const;
};

}  // namespace cstfem
