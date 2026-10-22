#include "cstfem/solver.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace cstfem {

double Reduction::norm(const std::vector<double>& a) const { return std::sqrt(dot(a, a)); }

void CsrOperator::apply(const std::vector<double>& x, std::vector<double>& y) const {
  A_.multiply(x, y);
}

double SerialReduction::dot(const std::vector<double>& a, const std::vector<double>& b) const {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

SolverStats pcg_jacobi(const LinearOperator& A, const std::vector<double>& inv_diag,
                       const Reduction& reduction, const std::vector<double>& b,
                       std::vector<double>& x, const SolverOptions& options) {
  const std::size_t n = b.size();
  SolverStats st;
  x.assign(n, 0.0);

  const double bnorm = reduction.norm(b);
  if (bnorm == 0.0) {
    st.converged = true;
    st.message = "zero right-hand side, x = 0";
    if (options.record_history) st.residual_history.push_back(0.0);
    return st;
  }

  std::vector<double> r = b, z(n), p(n), Ap(n);
  for (std::size_t i = 0; i < n; ++i) z[i] = inv_diag[i] * r[i];
  p = z;
  double rz = reduction.dot(r, z);
  st.rel_residual = 1.0;
  if (options.record_history) st.residual_history.push_back(1.0);

  for (int k = 1; k <= options.max_iter; ++k) {
    A.apply(p, Ap);
    const double pAp = reduction.dot(p, Ap);
    if (!(pAp > 0.0)) {
      std::ostringstream msg;
      msg << "CG breakdown at iteration " << k << ": p'Ap = " << pAp
          << " (matrix not SPD?)";
      st.message = msg.str();
      return st;
    }
    const double alpha = rz / pAp;
    for (std::size_t i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    st.iterations = k;
    st.rel_residual = reduction.norm(r) / bnorm;
    if (options.record_history) st.residual_history.push_back(st.rel_residual);
    if (st.rel_residual < options.rtol) {
      st.converged = true;
      std::ostringstream msg;
      msg << "converged in " << k << " iterations, relative residual " << st.rel_residual;
      st.message = msg.str();
      return st;
    }
    for (std::size_t i = 0; i < n; ++i) z[i] = inv_diag[i] * r[i];
    const double rz_new = reduction.dot(r, z);
    const double beta = rz_new / rz;
    rz = rz_new;
    for (std::size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
  }

  std::ostringstream msg;
  msg << "NOT converged: reached max_iter = " << options.max_iter << ", relative residual "
      << st.rel_residual << " (rtol " << options.rtol << ")";
  st.message = msg.str();
  return st;
}

void CgJacobiSolver::setup(const CsrMatrix& A) {
  A_ = &A;
  inv_diag_.resize(A.num_rows);
  for (int i = 0; i < A.num_rows; ++i) {
    const double d = A.diagonal(i);
    if (!(d > 0.0))
      throw std::runtime_error("cg_jacobi: non-positive diagonal at row " + std::to_string(i));
    inv_diag_[i] = 1.0 / d;
  }
}

SolverStats CgJacobiSolver::solve(const std::vector<double>& b, std::vector<double>& x) {
  if (!A_) throw std::logic_error("cg_jacobi: solve() called before setup()");
  stats_ = pcg_jacobi(CsrOperator(*A_), inv_diag_, SerialReduction(), b, x, options_);
  return stats_;
}

}  // namespace cstfem
