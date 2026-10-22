#pragma once

#include <string>
#include <vector>

#include "cstfem/csr.hpp"

namespace cstfem {

// y = A x. The distributed version (Milestone 4) replaces this with an operator that does
// the halo exchange.
class LinearOperator {
public:
  virtual ~LinearOperator() = default;
  virtual void apply(const std::vector<double>& x, std::vector<double>& y) const = 0;
};

// Inner products over the (possibly distributed) vector; MPI_Allreduce in Milestone 4.
class Reduction {
public:
  virtual ~Reduction() = default;
  virtual double dot(const std::vector<double>& a, const std::vector<double>& b) const = 0;
  double norm(const std::vector<double>& a) const;
};

class CsrOperator : public LinearOperator {
public:
  explicit CsrOperator(const CsrMatrix& A) : A_(A) {}
  void apply(const std::vector<double>& x, std::vector<double>& y) const override;

private:
  const CsrMatrix& A_;
};

class SerialReduction : public Reduction {
public:
  double dot(const std::vector<double>& a, const std::vector<double>& b) const override;
};

struct SolverOptions {
  double rtol = 1e-10;
  int max_iter = 20000;
  bool record_history = false;
};

struct SolverStats {
  int iterations = 0;
  double rel_residual = 0.0;       // recursive (updated) ||r_k|| / ||b|| at exit
  double true_rel_residual = 0.0;  // ||b - A x|| / ||b||, recomputed at exit
  bool converged = false;          // recursive residual reached rtol
  std::string message;             // reason for stopping, readable
  std::string warning;             // non-empty if converged but true residual > 10 rtol
  std::vector<double> residual_history;   // ||r_k|| / ||b||, k = 0.. (if recorded)
};

// Jacobi-preconditioned CG from x = 0. Stops when ||r_k|| / ||b|| < rtol; returns x = 0
// immediately if ||b|| = 0. Non-convergence (max_iter, breakdown) gives converged = false.
// On exit the true residual b - A x is recomputed; if it exceeds 10 rtol although the
// recursive residual converged, `warning` is set (the caller prints it; the library does not).
SolverStats pcg_jacobi(const LinearOperator& A, const std::vector<double>& inv_diag,
                       const Reduction& reduction, const std::vector<double>& b,
                       std::vector<double>& x, const SolverOptions& options);

class LinearSolver {
public:
  virtual ~LinearSolver() = default;
  virtual void setup(const CsrMatrix& A) = 0;
  // Solves A x = b; always check the returned stats (converged may be false).
  [[nodiscard]] virtual SolverStats solve(const std::vector<double>& b,
                                          std::vector<double>& x) = 0;
  virtual const SolverStats& stats() const = 0;
};

// Serial cg_jacobi backend.
class CgJacobiSolver : public LinearSolver {
public:
  explicit CgJacobiSolver(SolverOptions options = {}) : options_(options) {}
  void setup(const CsrMatrix& A) override;
  [[nodiscard]] SolverStats solve(const std::vector<double>& b, std::vector<double>& x) override;
  const SolverStats& stats() const override { return stats_; }

private:
  SolverOptions options_;
  const CsrMatrix* A_ = nullptr;
  std::vector<double> inv_diag_;
  SolverStats stats_;
};

}  // namespace cstfem
