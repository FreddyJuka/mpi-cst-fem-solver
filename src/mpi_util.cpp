#include "cstfem/mpi_util.hpp"

#include <climits>
#include <stdexcept>
#include <vector>

namespace cstfem {

std::string bcast_string(const std::string& s, int root, MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int len = rank == root ? static_cast<int>(s.size()) : 0;
  MPI_Bcast(&len, 1, MPI_INT, root, comm);
  std::vector<char> buf(s.begin(), s.end());
  buf.resize(len);
  MPI_Bcast(buf.data(), len, MPI_CHAR, root, comm);
  return std::string(buf.begin(), buf.end());
}

void check_collective(bool ok, const std::string& message, MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int failing = ok ? INT_MAX : rank;
  int first = INT_MAX;
  MPI_Allreduce(&failing, &first, 1, MPI_INT, MPI_MIN, comm);
  if (first == INT_MAX) return;
  throw std::runtime_error(bcast_string(message, first, comm));
}

}  // namespace cstfem
