#include <mpi.h>

#include <cstdio>
#include <iostream>
#include <string>

#include "cstfem/config.hpp"
#include "cstfem/driver.hpp"
#include "cstfem/parallel_driver.hpp"
#include "cstfem/version.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Arguments are identical on all ranks, so every rank takes the same branch.
  std::string case_path;
  bool partition_only = false, bad_args = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--partition-only")
      partition_only = true;
    else if (case_path.empty() && arg.rfind("--", 0) != 0)
      case_path = arg;
    else
      bad_args = true;
  }
  if (case_path.empty()) bad_args = true;

  if (rank == 0) {
    std::printf("mpi-cst-fem-solver %s\n", cstfem::version());
    std::printf("MPI ranks: %d\n", size);
    std::printf("Backends: METIS=yes ParMETIS=%s PETSc=%s\n", USE_PARMETIS ? "yes" : "no",
                USE_PETSC ? "yes" : "no");
    std::fflush(stdout);
  }

  int status = 0;
  if (bad_args) {
    if (rank == 0) std::fprintf(stderr, "usage: %s <case-file> [--partition-only]\n", argv[0]);
    status = 1;
  } else if (partition_only) {
    status = cstfem::run_partition_only(case_path, MPI_COMM_WORLD, std::cout, std::cerr);
  } else if (size > 1) {
    if (rank == 0)
      std::fprintf(stderr,
                   "error: launched with %d MPI ranks, but only the serial solver exists; "
                   "the parallel path arrives in Milestone 4. Run with 1 rank, or use "
                   "--partition-only.\n",
                   size);
    status = 1;
  } else {
    status = cstfem::run_serial_case(case_path, std::cout, std::cerr);
  }

  MPI_Finalize();
  return status;
}
