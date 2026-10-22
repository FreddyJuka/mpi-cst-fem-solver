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

// Distributed solve of a case file (np > 1): rank 0 reads the case (text broadcast) and the
// mesh; partition, distributed assembly, loads, Dirichlet, distributed Jacobi-PCG, element
// stresses and Kirsch metrics. Rank 0 prints iterations, residuals, error metrics and a load
// report. File output (VTU/CSV/timings) is not written yet. Collective; returns 0 on
// success, 2 if the solver did not converge, 1 on error.
int run_parallel_case(const std::string& case_path, MPI_Comm comm, std::ostream& out,
                      std::ostream& err);

}  // namespace cstfem
