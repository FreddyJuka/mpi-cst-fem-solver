// Clock consistency check (diagnostic, not part of the solver): samples MPI_Wtime,
// std::chrono::steady_clock, CLOCK_MONOTONIC and CLOCK_REALTIME together and reports their
// divergence and jumps. usage: clock_check [seconds=120] [interval_ms=100]
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <thread>

static double ts(clockid_t id) {
  timespec t;
  clock_gettime(id, &t);
  return double(t.tv_sec) + 1e-9 * double(t.tv_nsec);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  const double duration = argc > 1 ? std::atof(argv[1]) : 120.0;
  const int interval_ms = argc > 2 ? std::atoi(argv[2]) : 100;
  int *flag = nullptr, has = 0;
  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &flag, &has);
  std::printf("MPI_Wtick %.3e s, MPI_WTIME_IS_GLOBAL %s\n", MPI_Wtick(),
              has ? (*flag ? "true" : "false") : "unset");
  using SC = std::chrono::steady_clock;
  auto sc = [] { return std::chrono::duration<double>(SC::now().time_since_epoch()).count(); };
  const double w0 = MPI_Wtime(), s0 = sc(), m0 = ts(CLOCK_MONOTONIC), r0 = ts(CLOCK_REALTIME);
  double pw = w0, ps = s0, pm = m0, pr = r0;
  const char* names[4] = {"MPI_Wtime", "steady_clock", "CLOCK_MONOTONIC", "CLOCK_REALTIME"};
  double maxdiv[4] = {0, 0, 0, 0};  // divergence of each clock from CLOCK_MONOTONIC (elapsed)
  double maxstep[4] = {0, 0, 0, 0}, minstep[4] = {1e9, 1e9, 1e9, 1e9};
  int jumps[4] = {0, 0, 0, 0}, n = 0;
  const double expect = interval_ms * 1e-3;
  while (ts(CLOCK_MONOTONIC) - m0 < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    const double w = MPI_Wtime(), s = sc(), m = ts(CLOCK_MONOTONIC), r = ts(CLOCK_REALTIME);
    const double cur[4] = {w, s, m, r}, prev[4] = {pw, ps, pm, pr};
    const double base[4] = {w0, s0, m0, r0};
    for (int i = 0; i < 4; ++i) {
      const double step = cur[i] - prev[i];
      maxstep[i] = std::max(maxstep[i], step);
      minstep[i] = std::min(minstep[i], step);
      maxdiv[i] = std::max(maxdiv[i], std::fabs((cur[i] - base[i]) - (m - m0)));
      // a jump: the step differs from the step of the monotonic clock by more than 10 ms
      if (i != 2 && std::fabs(step - (m - pm)) > 0.010) {
        ++jumps[i];
        std::printf("jump: %s step %.6f s vs CLOCK_MONOTONIC %.6f s at t=%.2f s\n", names[i], step,
                    m - pm, m - m0);
      }
    }
    pw = w; ps = s; pm = m; pr = r;
    ++n;
  }
  std::printf("%d samples every %.0f ms over %.1f s (CLOCK_MONOTONIC)\n", n, expect * 1e3, pm - m0);
  for (int i = 0; i < 4; ++i)
    std::printf("%-16s elapsed %.6f s, max |divergence from CLOCK_MONOTONIC| %.6f s, step min/max "
                "%.4f/%.4f s, jumps > 10 ms: %d\n",
                names[i], (i == 0 ? pw - w0 : i == 1 ? ps - s0 : i == 2 ? pm - m0 : pr - r0),
                maxdiv[i], minstep[i], maxstep[i], jumps[i]);
  MPI_Finalize();
}
