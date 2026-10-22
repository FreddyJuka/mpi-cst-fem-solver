#include "cstfem/solver.hpp"

#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace cstfem {

double steady_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

double Reduction::norm(const std::vector<double>& a) const { return std::sqrt(dot(a, a)); }

void CsrOperator::apply(const std::vector<double>& x, std::vector<double>& y) const {
  BreakdownTimer tm(bd_);
  A_.multiply(x, y);
  tm.add(&SolveBreakdown::spmv_local);
  if (bd_) ++bd_->spmv_calls;
}

double SerialReduction::dot(const std::vector<double>& a, const std::vector<double>& b) const {
  BreakdownTimer tm(bd_);
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  tm.add(&SolveBreakdown::dot_local);
  if (bd_) ++bd_->reductions;
  return s;
}

namespace {

SolverStats pcg_iterate(const LinearOperator& A, const std::vector<double>& inv_diag,
                        const Reduction& reduction, const std::vector<double>& b,
                        std::vector<double>& x, const SolverOptions& options,
                        SolveBreakdown* timing) {
  const std::size_t n = b.size();
  SolverStats st;
  BreakdownTimer tm(timing);
  x.assign(n, 0.0);
  tm.add(&SolveBreakdown::vector_ops);

  const double bnorm = reduction.norm(b);
  if (bnorm == 0.0) {
    st.converged = true;
    st.message = "zero right-hand side, x = 0";
    if (options.record_history) st.residual_history.push_back(0.0);
    return st;
  }

  tm.restart();
  std::vector<double> r = b, z(n), p(n), Ap(n);
  for (std::size_t i = 0; i < n; ++i) z[i] = inv_diag[i] * r[i];
  p = z;
  tm.add(&SolveBreakdown::vector_ops);
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
    tm.restart();
    for (std::size_t i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    tm.add(&SolveBreakdown::vector_ops);
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
    tm.restart();
    for (std::size_t i = 0; i < n; ++i) z[i] = inv_diag[i] * r[i];
    tm.add(&SolveBreakdown::vector_ops);
    const double rz_new = reduction.dot(r, z);
    const double beta = rz_new / rz;
    rz = rz_new;
    tm.restart();
    for (std::size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    tm.add(&SolveBreakdown::vector_ops);
  }

  std::ostringstream msg;
  msg << "NOT converged: reached max_iter = " << options.max_iter << ", relative residual "
      << st.rel_residual << " (rtol " << options.rtol << ")";
  st.message = msg.str();
  return st;
}

}  // namespace

SolverStats pcg_jacobi(const LinearOperator& A, const std::vector<double>& inv_diag,
                       const Reduction& reduction, const std::vector<double>& b,
                       std::vector<double>& x, const SolverOptions& options,
                       SolveBreakdown* timing) {
  SolverStats st = pcg_iterate(A, inv_diag, reduction, b, x, options, timing);
  const double bnorm = reduction.norm(b);
  if (bnorm == 0.0) return st;

  std::vector<double> r;
  A.apply(x, r);
  BreakdownTimer tm(timing);
  for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - r[i];
  tm.add(&SolveBreakdown::vector_ops);
  st.true_rel_residual = reduction.norm(r) / bnorm;
  if (st.converged && st.true_rel_residual > 10.0 * options.rtol) {
    std::ostringstream msg;
    msg << "WARNING: CG recursive residual converged (" << st.rel_residual
        << ") but the true residual ||b - Ax|| / ||b|| = " << st.true_rel_residual
        << " exceeds 10 * rtol = " << 10.0 * options.rtol;
    st.warning = msg.str();
  }
  return st;
}

void CgJacobiSolver::setup(const CsrMatrix& A) {
  const double t0 = bd_ && bd_->clock ? bd_->clock() : 0.0;
  A_ = &A;
  inv_diag_.resize(A.num_rows);
  for (int i = 0; i < A.num_rows; ++i) {
    const double d = A.diagonal(i);
    if (!(d > 0.0))
      throw std::runtime_error("cg_jacobi: non-positive diagonal at row " + std::to_string(i));
    inv_diag_[i] = 1.0 / d;
  }
  if (bd_ && bd_->clock) {
    const double dt = bd_->clock() - t0;
    bd_->setup += dt;
    bd_->total += dt;
  }
}

SolverStats CgJacobiSolver::solve(const std::vector<double>& b, std::vector<double>& x) {
  if (!A_) throw std::logic_error("cg_jacobi: solve() called before setup()");
  const double t0 = bd_ && bd_->clock ? bd_->clock() : 0.0;
  stats_ = pcg_jacobi(CsrOperator(*A_, bd_), inv_diag_, SerialReduction(bd_), b, x, options_,
                      bd_);
  if (bd_ && bd_->clock) bd_->total += bd_->clock() - t0;
  return stats_;
}

}  // namespace cstfem
