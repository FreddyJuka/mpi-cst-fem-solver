#include <mpi.h>

#include <cstdio>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int one = 1;
  int sum = 0;
  MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int ok = (sum == size);
  int all_ok = 0;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("mpi_smoke: sum=%d size=%d -> %s\n", sum, size, all_ok ? "PASS" : "FAIL");
  }

  MPI_Finalize();
  return all_ok ? 0 : 1;
}
