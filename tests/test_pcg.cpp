// Jacobi-PCG against a dense Cholesky solution, plus edge cases.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "cstfem/solver.hpp"
#include "test_assert.hpp"

using namespace cstfem;

namespace {

using Dense = std::vector<double>;  // row-major n x n

CsrMatrix dense_to_csr(const Dense& A, int n) {
  CsrMatrix K;
  K.num_rows = K.num_cols = n;
  K.row_ptr.push_back(0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (A[i * n + j] != 0.0) {
        K.cols.push_back(j);
        K.vals.push_back(A[i * n + j]);
      }
    }
    K.row_ptr.push_back(static_cast<int>(K.cols.size()));
  }
  return K;
}

// Solves A x = b by Cholesky A = L L^T.
std::vector<double> cholesky_solve(Dense A, int n, std::vector<double> b) {
  for (int j = 0; j < n; ++j) {
    double d = A[j * n + j];
    for (int k = 0; k < j; ++k) d -= A[j * n + k] * A[j * n + k];
    A[j * n + j] = std::sqrt(d);
    for (int i = j + 1; i < n; ++i) {
      double s = A[i * n + j];
      for (int k = 0; k < j; ++k) s -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s / A[j * n + j];
    }
  }
  for (int i = 0; i < n; ++i) {  // L y = b
    for (int k = 0; k < i; ++k) b[i] -= A[i * n + k] * b[k];
    b[i] /= A[i * n + i];
  }
  for (int i = n - 1; i >= 0; --i) {  // L^T x = y
    for (int k = i + 1; k < n; ++k) b[i] -= A[k * n + i] * b[k];
    b[i] /= A[i * n + i];
  }
  return b;
}

// Random SPD matrix M^T M + n I with a spread-out diagonal scaling.
Dense random_spd(int n, std::mt19937& gen) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Dense M(n * n), A(n * n, 0.0);
  for (double& v : M) v = u(gen);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) A[i * n + j] += M[k * n + i] * M[k * n + j];
      if (i == j) A[i * n + j] += n;
    }
  // D A D keeps SPD and makes Jacobi scaling matter.
  std::vector<double> d(n);
  for (int i = 0; i < n; ++i) d[i] = std::pow(10.0, 3.0 * (u(gen) + 1.0) / 2.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) A[i * n + j] *= d[i] * d[j];
  return A;
}

void test_random_spd() {
  std::mt19937 gen(2026);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int n : {5, 40, 120}) {
    const Dense A = random_spd(n, gen);
    std::vector<double> b(n);
    for (double& v : b) v = u(gen);
    const auto x_ref = cholesky_solve(A, n, b);

    const CsrMatrix K = dense_to_csr(A, n);
    CgJacobiSolver solver({1e-12, 1000, true});
    solver.setup(K);
    std::vector<double> x;
    const SolverStats st = solver.solve(b, x);
    double err = 0.0, ref = 0.0;
    for (int i = 0; i < n; ++i) {
      err = std::max(err, std::abs(x[i] - x_ref[i]));
      ref = std::max(ref, std::abs(x_ref[i]));
    }
    std::printf("  n = %3d: %d iterations, rel. residual %.2e, max rel. error vs Cholesky %.2e\n",
                n, st.iterations, st.rel_residual, err / ref);
    CHECK(st.converged);
    CHECK(st.rel_residual < 1e-12);
    CHECK(err / ref < 1e-9);
    CHECK(st.residual_history.size() == static_cast<std::size_t>(st.iterations) + 1);
    CHECK(st.residual_history.front() == 1.0 && st.residual_history.back() < 1e-12);
  }
}

void test_zero_rhs() {
  std::mt19937 gen(1);
  const CsrMatrix K = dense_to_csr(random_spd(6, gen), 6);
  CgJacobiSolver solver;
  solver.setup(K);
  std::vector<double> x(6, 123.0);
  const SolverStats st = solver.solve(std::vector<double>(6, 0.0), x);
  CHECK(st.converged && st.iterations == 0);
  CHECK(std::all_of(x.begin(), x.end(), [](double v) { return v == 0.0; }));
}

void test_not_converged() {
  std::mt19937 gen(3);
  const int n = 30;
  const CsrMatrix K = dense_to_csr(random_spd(n, gen), n);
  CgJacobiSolver solver({1e-14, 2, false});
  solver.setup(K);
  std::vector<double> x;
  const SolverStats st = solver.solve(std::vector<double>(n, 1.0), x);
  std::printf("  max_iter = 2: %s\n", st.message.c_str());
  CHECK(!st.converged);
  CHECK(st.iterations == 2);
  CHECK(st.message.find("NOT converged") != std::string::npos);
  CHECK(!solver.stats().converged);
}

void test_bad_diagonal() {
  CsrMatrix K = dense_to_csr({2.0, 1.0, 1.0, -1.0}, 2);
  CgJacobiSolver solver;
  CHECK_THROWS(solver.setup(K), "non-positive diagonal");
}

}  // namespace

int main() {
  RUN_TEST(test_random_spd);
  RUN_TEST(test_zero_rhs);
  RUN_TEST(test_not_converged);
  RUN_TEST(test_bad_diagonal);
  return test_summary();
}
