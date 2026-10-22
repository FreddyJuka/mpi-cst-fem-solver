#pragma once

#include <mpi.h>

#include <ostream>
#include <string>

namespace cstfem {

// `cstfem_solve <case> --partition-only`: rank 0 reads the case and the mesh, partitions it
// for all ranks of `comm`, distributes it, prints the partition report, and every rank writes
// its piece of <output_dir>/partition.pvtu. Works with any number of ranks. Collective;
// returns 0 on success, 1 on error (message printed by rank 0 on `err`).
int run_partition_only(const std::string& case_path, MPI_Comm comm, std::ostream& out,
                       std::ostream& err);

// `cstfem_solve <case>`: selects the code path from the case key `path` (auto: serial for 1
// rank, distributed otherwise) and runs it. The distributed path: rank 0 reads the case (text
// broadcast) and the mesh; partition (METIS or ParMETIS), distribution, distributed assembly,
// loads, Dirichlet, Jacobi-PCG, element stresses, Kirsch metrics; every rank writes its VTU
// piece, rank 0 the .pvtu, the gathered CSVs and timings.json (per-phase min/max/mean over
// ranks, load report). Collective; returns 0 on success, 2 if the solver did not converge,
// 1 on error.
int run_case(const std::string& case_path, MPI_Comm comm, std::ostream& out, std::ostream& err);

}  // namespace cstfem
