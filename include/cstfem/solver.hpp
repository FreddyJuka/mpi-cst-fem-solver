#pragma once

#include <string>
#include <vector>

#include "cstfem/csr.hpp"

namespace cstfem {

// Optional per-rank wall-clock accounting of a solve. Timing is active only when a
// SolveBreakdown with a non-null clock is passed; the arithmetic and the communication are
// identical either way. `clock` returns seconds: steady_seconds() in the serial code,
// MPI_Wtime in the distributed code.
using WallClock = double (*)();
double steady_seconds();

struct SolveBreakdown {
  WallClock clock = nullptr;
  double setup = 0.0;       // Jacobi diagonal (+ its collective check in parallel)
  double spmv_local = 0.0;  // A_loc x (serial: the whole SpMV)
  double halo_post = 0.0;   // pack + MPI_Startall, and unpack after the wait
  double halo_wait = 0.0;   // MPI_Waitall of the halo exchange
  double spmv_ghost = 0.0;  // A_ghost x_halo
  double dot_local = 0.0;   // local part of dot products and norms
  double allreduce = 0.0;   // MPI_Allreduce of dot products and norms
  double vector_ops = 0.0;  // axpy updates, preconditioner apply, b - Ax, work vectors
  double total = 0.0;       // measured around the whole solve (setup + PCG + true residual)
  long long spmv_calls = 0, reductions = 0;
  long long halo_bytes_sent = 0, halo_messages_sent = 0;

  double accounted() const {
    return setup + spmv_local + halo_post + halo_wait + spmv_ghost + dot_local + allreduce +
           vector_ops;
  }
};

// Adds the time since the last restart()/add() to a SolveBreakdown field; no-op when off.
class BreakdownTimer {
public:
  explicit BreakdownTimer(SolveBreakdown* bd) : bd_(bd && bd->clock ? bd : nullptr) {
    restart();
  }
  void restart() {
    if (bd_) t0_ = bd_->clock();
  }
  void add(double SolveBreakdown::*field) {
    if (!bd_) return;
    const double t = bd_->clock();
    bd_->*field += t - t0_;
    t0_ = t;
  }

private:
  SolveBreakdown* bd_;
  double t0_ = 0.0;
};

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
  explicit CsrOperator(const CsrMatrix& A, SolveBreakdown* bd = nullptr) : A_(A), bd_(bd) {}
  void apply(const std::vector<double>& x, std::vector<double>& y) const override;

private:
  const CsrMatrix& A_;
  SolveBreakdown* bd_;
};

class SerialReduction : public Reduction {
public:
  explicit SerialReduction(SolveBreakdown* bd = nullptr) : bd_(bd) {}
  double dot(const std::vector<double>& a, const std::vector<double>& b) const override;

private:
  SolveBreakdown* bd_;
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
// `timing` (optional) accumulates the vector-operation time; the operator and the reduction
// account for their own parts.
SolverStats pcg_jacobi(const LinearOperator& A, const std::vector<double>& inv_diag,
                       const Reduction& reduction, const std::vector<double>& b,
                       std::vector<double>& x, const SolverOptions& options,
                       SolveBreakdown* timing = nullptr);

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
  // Optional timing of setup() and solve() (set before setup()); nullptr turns it off.
  void set_breakdown(SolveBreakdown* bd) { bd_ = bd; }

private:
  SolveBreakdown* bd_ = nullptr;
  SolverOptions options_;
  const CsrMatrix* A_ = nullptr;
  std::vector<double> inv_diag_;
  SolverStats stats_;
};

}  // namespace cstfem
