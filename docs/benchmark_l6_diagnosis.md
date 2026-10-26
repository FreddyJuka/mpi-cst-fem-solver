# L6 p = 1 bimodality diagnosis: raw measurements (Milestone 7c)

Produced by `python/l6_diagnosis.sh` (11 runs, one after another, same machine and session). Per run: `free -m` before the run, `vmstat 1` and a 1 s `ps` RSS sampler during it, `/usr/bin/time -v` around `mpiexec --bind-to core --map-by core`. Elapsed is the wall time of `mpiexec` as measured by `/usr/bin/time`; "in-program total" is the `total` phase (max over ranks) printed by the solver. The two differ by at most 0.7 s (either sign) for GAMG, but for the Jacobi runs the in-program total is larger than the elapsed wall time by 14.9, 14.1 and 11.5 s (2.5-3.2 %); this was not investigated. Conclusions below use elapsed wall time for the diagnosis and the in-program `total` only for comparison with 7b. Max RSS is `ru_maxrss` of the largest single process (per rank for p = 4); for p = 4 the peak sum over ranks of the sampled RSS is also given. si/so are the sums of the `vmstat` swap-in/swap-out columns over the run; "min free" is the lowest `free` column of `vmstat` during the run. The host has 7,749 MiB RAM and 2,048 MiB swap (swap used before every run: 0).

| run | elapsed [s] | in-program total [s] | max RSS [MiB] | peak sum RSS [MiB] | major faults | minor faults | swaps | sum si | sum so | free before [MiB] | min free during [MiB] | page cache before [MiB] |
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
