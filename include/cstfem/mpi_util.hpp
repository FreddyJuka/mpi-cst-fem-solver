#pragma once

#include <mpi.h>

#include <string>

namespace cstfem {

// Broadcasts a string from `root`.
std::string bcast_string(const std::string& s, int root, MPI_Comm comm);

// Collective error check: if `ok` is false on any rank, every rank throws
// std::runtime_error carrying the message of the lowest failing rank. Use it after work
// that may fail on some ranks only, so the others do not hang in the next collective.
void check_collective(bool ok, const std::string& message, MPI_Comm comm);

}  // namespace cstfem
