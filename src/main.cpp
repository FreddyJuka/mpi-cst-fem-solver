#include <mpi.h>

#include <cstdio>
#include <iostream>

#include "cstfem/config.hpp"
#include "cstfem/driver.hpp"
#include "cstfem/version.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int status = 0;
  if (rank == 0) {
    std::printf("mpi-cst-fem-solver %s\n", cstfem::version());
    std::printf("MPI ranks: %d\n", size);
    std::printf("Backends: METIS=yes ParMETIS=%s PETSc=%s\n", USE_PARMETIS ? "yes" : "no",
                USE_PETSC ? "yes" : "no");
    std::fflush(stdout);
    if (argc != 2) {
      std::fprintf(stderr, "usage: %s <case-file>\n", argv[0]);
      status = 1;
    } else if (size > 1) {
      std::fprintf(stderr,
                   "error: launched with %d MPI ranks, but only the serial solver exists; "
                   "the parallel path arrives in Milestone 4. Run with 1 rank.\n",
                   size);
      status = 1;
    } else {
      status = cstfem::run_serial_case(argv[1], std::cout, std::cerr);
    }
  }
  MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Finalize();
  return status;
}
