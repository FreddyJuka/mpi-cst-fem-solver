#include <mpi.h>

#include <cstdio>

#include "cstfem/config.hpp"
#include "cstfem/version.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::printf("mpi-cst-fem-solver %s\n", cstfem::version());
    std::printf("MPI ranks: %d\n", size);
    std::printf("Backends: METIS=yes ParMETIS=%s PETSc=%s\n", USE_PARMETIS ? "yes" : "no",
                USE_PETSC ? "yes" : "no");
  }

  MPI_Finalize();
  return 0;
}
