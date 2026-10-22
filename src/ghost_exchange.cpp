#include "cstfem/ghost_exchange.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cstfem {

namespace {

constexpr int kForwardTag = 101;
constexpr int kReverseTag = 102;

int owner_of(const std::vector<int>& offsets, int global) {
  return static_cast<int>(std::upper_bound(offsets.begin(), offsets.end(), global) -
                          offsets.begin()) -
         1;
}

}  // namespace

GhostExchange::GhostExchange(std::vector<int> ghost_globals, const std::vector<int>& offsets,
                             MPI_Comm comm)
    : ghost_globals_(std::move(ghost_globals)) {
  // Own communicator: messages of different exchanges can never match each other.
  MPI_Comm_dup(comm, &comm_);
  int rank = 0, size = 1;
  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);

  // Ghost positions grouped by owner.
  std::vector<std::vector<int>> by_owner(size);
  for (int k = 0; k < num_ghosts(); ++k) {
    const int g = ghost_globals_[k];
    const int o = owner_of(offsets, g);
    if (o < 0 || o >= size || o == rank)
      throw std::logic_error("GhostExchange: ghost " + std::to_string(g) +
                             " has invalid owner " + std::to_string(o));
    by_owner[o].push_back(k);
  }
  std::vector<int> recv_counts(size), recv_displs(size), send_counts(size), send_displs(size);
  std::vector<int> requested;
  for (int q = 0; q < size; ++q) {
    recv_displs[q] = static_cast<int>(recv_pos_.size());
    recv_counts[q] = static_cast<int>(by_owner[q].size());
    for (int k : by_owner[q]) {
      recv_pos_.push_back(k);
      requested.push_back(ghost_globals_[k]);
    }
  }

  // Tell each owner which of its entries we need.
  MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, comm_);
  int nsend = 0;
  for (int q = 0; q < size; ++q) {
    send_displs[q] = nsend;
    nsend += send_counts[q];
  }
  std::vector<int> asked(nsend);
  MPI_Alltoallv(requested.data(), recv_counts.data(), recv_displs.data(), MPI_INT, asked.data(),
                send_counts.data(), send_displs.data(), MPI_INT, comm_);
  send_index_.resize(nsend);
  const int first = offsets[rank], n_owned = offsets[rank + 1] - offsets[rank];
  for (int i = 0; i < nsend; ++i) {
    send_index_[i] = asked[i] - first;
    if (send_index_[i] < 0 || send_index_[i] >= n_owned)
      throw std::logic_error("GhostExchange: rank asked for entry " + std::to_string(asked[i]) +
                             " that it does not own");
  }

  // Persistent requests. Buffers are sized once and never reallocated.
  fwd_send_.resize(nsend);
  rev_recv_.resize(nsend);
  fwd_recv_.resize(recv_pos_.size());
  rev_send_.resize(recv_pos_.size());
  for (int q = 0; q < size; ++q) {
    if (send_counts[q] > 0 || recv_counts[q] > 0) ++num_neighbors_;
    if (send_counts[q] > 0) {
      MPI_Request r;
      MPI_Send_init(fwd_send_.data() + send_displs[q], send_counts[q], MPI_DOUBLE, q,
                    kForwardTag, comm_, &r);
      fwd_requests_.push_back(r);
      MPI_Recv_init(rev_recv_.data() + send_displs[q], send_counts[q], MPI_DOUBLE, q,
                    kReverseTag, comm_, &r);
      rev_requests_.push_back(r);
    }
    if (recv_counts[q] > 0) {
      MPI_Request r;
      MPI_Recv_init(fwd_recv_.data() + recv_displs[q], recv_counts[q], MPI_DOUBLE, q,
                    kForwardTag, comm_, &r);
      fwd_requests_.push_back(r);
      MPI_Send_init(rev_send_.data() + recv_displs[q], recv_counts[q], MPI_DOUBLE, q,
                    kReverseTag, comm_, &r);
      rev_requests_.push_back(r);
    }
  }
}

GhostExchange::~GhostExchange() {
  for (auto& r : fwd_requests_) MPI_Request_free(&r);
  for (auto& r : rev_requests_) MPI_Request_free(&r);
  if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
}

void GhostExchange::begin_forward(const double* owned) {
  for (std::size_t i = 0; i < send_index_.size(); ++i) fwd_send_[i] = owned[send_index_[i]];
  if (!fwd_requests_.empty())
    MPI_Startall(static_cast<int>(fwd_requests_.size()), fwd_requests_.data());
}

void GhostExchange::end_forward(double* ghosts) {
  if (!fwd_requests_.empty())
    MPI_Waitall(static_cast<int>(fwd_requests_.size()), fwd_requests_.data(),
                MPI_STATUSES_IGNORE);
  for (std::size_t i = 0; i < recv_pos_.size(); ++i) ghosts[recv_pos_[i]] = fwd_recv_[i];
}

void GhostExchange::forward(const double* owned, double* ghosts) {
  begin_forward(owned);
  end_forward(ghosts);
}

const std::vector<double>& GhostExchange::reverse(const double* ghosts) {
  for (std::size_t i = 0; i < recv_pos_.size(); ++i) rev_send_[i] = ghosts[recv_pos_[i]];
  if (!rev_requests_.empty()) {
    MPI_Startall(static_cast<int>(rev_requests_.size()), rev_requests_.data());
    MPI_Waitall(static_cast<int>(rev_requests_.size()), rev_requests_.data(),
                MPI_STATUSES_IGNORE);
  }
  return rev_recv_;
}

void GhostExchange::reverse_add(const double* ghosts, double* owned) {
  const auto& received = reverse(ghosts);
  for (std::size_t i = 0; i < received.size(); ++i) owned[send_index_[i]] += received[i];
}

}  // namespace cstfem
