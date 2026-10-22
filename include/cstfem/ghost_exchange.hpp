#pragma once

#include <mpi.h>

#include <vector>

namespace cstfem {

// Communication pattern between owners and ghost copies of distributed entries (DOFs).
//
// Entries are owned in contiguous global ranges given by `offsets` (rank r owns
// [offsets[r], offsets[r + 1])). A rank holds `ghost_globals[k]` as ghost k. The pattern is
// built once (MPI_Alltoall of counts + MPI_Alltoallv of the requested global ids) and the
// exchanges use persistent point-to-point requests on neighbours only:
//   forward: owner value -> every ghost copy               (e.g. SpMV halo, solution)
//   reverse: ghost contributions -> owner, combined there  (e.g. loads, Dirichlet flags)
// Construction and every exchange are collective over `comm`.
class GhostExchange {
public:
  GhostExchange(std::vector<int> ghost_globals, const std::vector<int>& offsets, MPI_Comm comm);
  ~GhostExchange();
  GhostExchange(const GhostExchange&) = delete;
  GhostExchange& operator=(const GhostExchange&) = delete;

  int num_ghosts() const { return static_cast<int>(ghost_globals_.size()); }
  int num_neighbors() const { return num_neighbors_; }
  const std::vector<int>& ghost_globals() const { return ghost_globals_; }

  // ghosts[k] <- owner's value of ghost_globals[k]; `owned` is indexed by owned local index
  // (global id - offsets[rank]). Split so that local work can overlap the communication.
  void begin_forward(const double* owned);
  void end_forward(double* ghosts);
  void forward(const double* owned, double* ghosts);

  // Sends ghosts[k] to the owner. Returns the received contributions; entry i belongs to the
  // owned local index reverse_targets()[i] (several entries may target the same index).
  const std::vector<double>& reverse(const double* ghosts);
  const std::vector<int>& reverse_targets() const { return send_index_; }
  // owned[t] += every contribution received for t.
  void reverse_add(const double* ghosts, double* owned);

private:
  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<int> ghost_globals_;
  int num_neighbors_ = 0;

  // Forward direction: send owned values at send_index_ (grouped by destination rank),
  // receive into ghost positions recv_pos_ (grouped by source rank).
  std::vector<int> send_index_, recv_pos_;
  std::vector<double> fwd_send_, fwd_recv_, rev_send_, rev_recv_;
  std::vector<MPI_Request> fwd_requests_, rev_requests_;
};

}  // namespace cstfem
