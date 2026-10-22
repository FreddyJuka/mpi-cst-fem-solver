add_executable(test_mpi_smoke test_mpi_smoke.cpp)
target_link_libraries(test_mpi_smoke PRIVATE MPI::MPI_CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_mpi_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Serial unit tests.
foreach(t test_bc test_case test_csr test_element test_mesh_reader test_patch test_pcg)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE cstfem_core)
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
  if(CSTFEM_SANITIZE)
    # Leak detection stays on; only leaks inside the MPI stack are suppressed. The slow
    # unwinder is needed because MPI libraries are built without frame pointers, so fast
    # unwinding truncates stacks before they reach the suppressed libraries.
    set_tests_properties(mpi_smoke_np${n} PROPERTIES ENVIRONMENT
      "ASAN_OPTIONS=fast_unwind_on_malloc=0;LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan.supp:fast_unwind_on_malloc=0")
  endif()
endforeach()
