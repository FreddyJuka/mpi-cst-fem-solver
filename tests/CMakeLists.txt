add_executable(test_mpi_smoke test_mpi_smoke.cpp)
target_link_libraries(test_mpi_smoke PRIVATE MPI::MPI_CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_mpi_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(_mpi_extra ${MPIEXEC_PREFLAGS})
if(CSTFEM_MPI_OVERSUBSCRIBE)
  list(APPEND _mpi_extra --oversubscribe)
endif()

foreach(n 1 2 4)
  add_test(NAME mpi_smoke_np${n}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${_mpi_extra}
            $<TARGET_FILE:test_mpi_smoke> ${MPIEXEC_POSTFLAGS})
endforeach()
