#!/bin/bash
# Diagnosis of the bimodal L6 GAMG p = 1 time (docs/BENCHMARKS.md, "L6 p = 1 bimodality").
# Per run: `free -m` before, `vmstat 1` and a 1 s `ps` RSS sampler in parallel, /usr/bin/time -v.
# usage: python/l6_diagnosis.sh OUTDIR     (run from the repository root; needs build/ and build-gamg/)
set -u
OUT=${1:?outdir}; mkdir -p "$OUT"
sed "s#^output_dir.*#output_dir       = $OUT/out#" cases/scaling_gamg_strong_L6.case > "$OUT/gamg.case"
sed "s#^output_dir.*#output_dir       = $OUT/out#" cases/scaling_strong_L6.case > "$OUT/jac.case"
one() {  # label np exe case
  local label=$1 np=$2 exe=$3 case=$4
  free -m > "$OUT/$label.free"; rm -rf "$OUT/out"
  vmstat -n 1 > "$OUT/$label.vmstat" & local vp=$!
  ( while true; do ps -C cstfem_solve -o rss= | awk '{s+=$1} END{print s+0}'; sleep 1; done ) > "$OUT/$label.rsssum" & local rp=$!
  /usr/bin/time -v -o "$OUT/$label.time" mpiexec --bind-to core --map-by core -n "$np" "$exe" "$case" \
      > "$OUT/$label.stdout" 2> "$OUT/$label.stderr"
  kill $vp $rp; wait $vp $rp 2>/dev/null
}
for i in 1 2 3 4 5; do one gamg1_$i 1 build-gamg/cstfem_solve "$OUT/gamg.case"; done
for i in 1 2 3;     do one gamg4_$i 4 build-gamg/cstfem_solve "$OUT/gamg.case"; done
for i in 1 2 3;     do one jac1_$i  1 build/cstfem_solve      "$OUT/jac.case";  done
