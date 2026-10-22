// STREAM-like triad a[i] = b[i] + s * c[i] for estimating memory bandwidth (Milestone 6a).
//
// Every MPI rank runs the triad on its own three arrays at the same time (MPI_Barrier before
// each repetition), so the aggregate bandwidth for 1, 2, 4, ... ranks shows where the memory
// system saturates, i.e. where a bandwidth-bound kernel such as the SpMV stops scaling.
// Run without mpiexec for a single-threaded measurement.
//
// Bytes counted per element: 24 (read b and c, write a), the STREAM convention; with
// write-allocate caches the actual traffic is 32 bytes per element.
//
// Usage: stream_triad [elements per array, default 2^24] [repetitions, default 20]

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const long n = argc > 1 ? std::atol(argv[1]) : (1L << 24);
  const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
  const double scalar = 3.0;

  std::vector<double> a(n, 0.0), b(n, 1.0), c(n, 2.0);  // first touch by this rank
  double best = 1e300, sum_time = 0.0;
  for (int r = 0; r < reps; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (long i = 0; i < n; ++i) a[i] = b[i] + scalar * c[i];
    const double mine = MPI_Wtime() - t0;
    double slowest = 0.0;  // the repetition ends when the slowest rank is done
    MPI_Allreduce(&mine, &slowest, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (r > 0) {  // the first repetition is a warm-up
      best = std::min(best, slowest);
      sum_time += slowest;
    }
  }

  // Check the result so the loop cannot be optimized away.
  const double expect = 1.0 + scalar * 2.0;
  long bad = 0;
  for (long i = 0; i < n; i += 4099) bad += a[i] != expect;
  long bad_all = 0;
  MPI_Reduce(&bad, &bad_all, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const double bytes = 24.0 * static_cast<double>(n) * size;
    const double best_gbs = bytes / best / 1e9, avg_gbs = bytes / (sum_time / (reps - 1)) / 1e9;
    std::printf("ranks %d, %ld elements per array (%.0f MiB per rank), %d repetitions: "
                "aggregate %.2f GB/s best, %.2f GB/s average; per rank %.2f GB/s best%s\n",
                size, n, 3.0 * 8.0 * static_cast<double>(n) / (1 << 20), reps, best_gbs, avg_gbs,
                best_gbs / size, bad_all ? "  [RESULT CHECK FAILED]" : "");
  }
  MPI_Finalize();
  return bad_all ? 1 : 0;
}
