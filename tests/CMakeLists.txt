add_executable(test_mpi_smoke test_mpi_smoke.cpp)
target_link_libraries(test_mpi_smoke PRIVATE MPI::MPI_CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_mpi_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Serial unit tests.
foreach(t test_case test_element test_mesh_reader)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE cstfem)
  target_compile_definitions(${t} PRIVATE
    CSTFEM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    CSTFEM_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}")
  add_test(NAME ${t} COMMAND ${t})
endforeach()

set(_mpi_extra ${MPIEXEC_PREFLAGS})
if(CSTFEM_MPI_OVERSUBSCRIBE)
  list(APPEND _mpi_extra --oversubscribe)
endif()

foreach(n 1 2 4)
  add_test(NAME mpi_smoke_np${n}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${_mpi_extra}
            $<TARGET_FILE:test_mpi_smoke> ${MPIEXEC_POSTFLAGS})
endforeach()
