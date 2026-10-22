#include "cstfem/parallel_driver.hpp"

#include <stdexcept>

#include "cstfem/case.hpp"
#include "cstfem/driver.hpp"
#include "cstfem/mpi_util.hpp"
#include "cstfem/partition.hpp"

namespace cstfem {

int run_partition_only(const std::string& case_path, MPI_Comm comm, std::ostream& out,
                       std::ostream& err) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  try {
    // Only rank 0 reads files; the paths are broadcast.
    std::string mesh_path, output_dir, error;
    bool ok = true;
    if (rank == 0) {
      try {
        const CaseFile c = CaseFile::load(case_path);
        c.warn_unknown_keys(known_case_keys(), err);
        mesh_path = c.get_string("mesh");
        output_dir = c.get_string("output_dir", "results");
      } catch (const std::exception& e) {
        ok = false;
        error = e.what();
      }
    }
    check_collective(ok, error, comm);
    mesh_path = bcast_string(mesh_path, 0, comm);
    output_dir = bcast_string(output_dir, 0, comm);

    PartitionSummary summary;
    const DistributedMesh dm = read_partition_distribute(mesh_path, comm, &summary);
    print_partition_report(dm, summary, comm, out);
    write_partition_vtu(dm, output_dir, "partition", comm);
    if (rank == 0)
      out << "wrote " << output_dir << "/partition.pvtu (" << dm.nranks << " piece(s))\n";
    return 0;
  } catch (const std::exception& e) {
    // Errors are collective (check_collective), so every rank gets here together.
    if (rank == 0) err << "error: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace cstfem
