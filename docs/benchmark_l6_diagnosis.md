# L6 p = 1 bimodality diagnosis: raw measurements (Milestone 7c)

Produced by `python/l6_diagnosis.sh` (11 runs, one after another, same machine and session). Per run: `free -m` before the run, `vmstat 1` and a 1 s `ps` RSS sampler during it, `/usr/bin/time -v` around `mpiexec --bind-to core --map-by core`. **Elapsed** is the wall time of `mpiexec` as reported by GNU `/usr/bin/time`, which reads `CLOCK_REALTIME`. Under WSL2 that clock is stepped backward by about 0.4-0.6 s every 32 s (see "Timing methodology" in [`BENCHMARKS.md`](BENCHMARKS.md)), so these elapsed values are **biased low**, by up to about 0.4 s for the 20 s GAMG runs and by 11.5-14.9 s (2.5-3.2 %) for the 420-480 s Jacobi runs, where the in-program total exceeds them. **In-program total** is the `total` phase (max over ranks) printed by the solver, measured with `MPI_Wtime` (monotonic); it excludes MPI start-up and exit and is the reliable duration. The swap/fault conclusions do not depend on either column. Max RSS is `ru_maxrss` of the largest single process (per rank for p = 4); for p = 4 the peak sum over ranks of the sampled RSS is also given. si/so are the sums of the `vmstat` swap-in/swap-out columns over the run; "min free" is the lowest `free` column of `vmstat` during the run. The host has 7,749 MiB RAM and 2,048 MiB swap (swap used before every run: 0).

| run | elapsed, GNU time (CLOCK_REALTIME, biased low) [s] | in-program total (MPI_Wtime) [s] | max RSS [MiB] | peak sum RSS [MiB] | major faults | minor faults | swaps | sum si | sum so | free before [MiB] | min free during [MiB] | page cache before [MiB] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L6 GAMG p = 1 #1 | 21.7 | 22.1 | 2048 | 2048 | 818 | 519592 | 0 | 0 | 0 | 5826 | 3602 | 815 |
| L6 GAMG p = 1 #2 | 20.4 | 21.1 | 2048 | 2048 | 5 | 520083 | 0 | 0 | 0 | 5371 | 3470 | 1258 |
| L6 GAMG p = 1 #3 | 22.5 | 22.0 | 2048 | 2048 | 5 | 520075 | 0 | 0 | 0 | 5364 | 3477 | 1258 |
| L6 GAMG p = 1 #4 | 21.8 | 22.4 | 2048 | 2048 | 5 | 520081 | 0 | 0 | 0 | 5355 | 3492 | 1259 |
| L6 GAMG p = 1 #5 | 23.2 | 23.8 | 2048 | 2048 | 5 | 520084 | 0 | 0 | 0 | 5335 | 3297 | 1259 |
| L6 GAMG p = 4 #1 | 16.3 | 15.8 | 702 | 2481 | 21 | 612909 | 0 | 0 | 0 | 5339 | 3048 | 1285 |
| L6 GAMG p = 4 #2 | 17.4 | 18.0 | 702 | 2488 | 65 | 613087 | 0 | 0 | 0 | 5329 | 3257 | 1287 |
| L6 GAMG p = 4 #3 | 17.3 | 16.8 | 702 | 2485 | 53 | 612857 | 0 | 0 | 0 | 5330 | 3268 | 1287 |
| L6 in-house Jacobi p = 1 #1 | 463.1 | 478.0 | 1657 | 1518 | 12 | 922349 | 0 | 0 | 0 | 5328 | 4084 | 1288 |
| L6 in-house Jacobi p = 1 #2 | 464.1 | 478.2 | 1657 | 1416 | 5 | 922346 | 0 | 0 | 0 | 5271 | 4187 | 1320 |
| L6 in-house Jacobi p = 1 #3 | 418.7 | 430.2 | 1657 | 1398 | 5 | 922351 | 0 | 0 | 0 | 5267 | 4169 | 1332 |
