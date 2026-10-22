#include "cstfem/csr.hpp"

#include <algorithm>

namespace cstfem {

int CsrMatrix::find(int row, int col) const {
  const auto begin = cols.begin() + row_ptr[row];
  const auto end = cols.begin() + row_ptr[row + 1];
  const auto it = std::lower_bound(begin, end, col);
  return (it != end && *it == col) ? static_cast<int>(it - cols.begin()) : -1;
}

double CsrMatrix::diagonal(int row) const {
  const int k = find(row, first_row + row);
  return k >= 0 ? vals[k] : 0.0;
}

void CsrMatrix::multiply(const std::vector<double>& x, std::vector<double>& y) const {
  y.resize(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    double s = 0.0;
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) s += vals[k] * x[cols[k]];
    y[i] = s;
  }
}

}  // namespace cstfem
