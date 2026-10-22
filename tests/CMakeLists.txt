# ---- MPI launcher settings (shared by MPI and end-to-end tests)
set(_mpi_extra ${MPIEXEC_PREFLAGS})
if(CSTFEM_MPI_OVERSUBSCRIBE)
  list(APPEND _mpi_extra --oversubscribe)
endif()
set(_mpi_env "")
if(CSTFEM_SANITIZE)
  # Leak detection stays on; only leaks inside the MPI stack are suppressed. The slow
  # unwinder is needed because MPI libraries are built without frame pointers, so fast
  # unwinding truncates stacks before they reach the suppressed libraries.
  set(_mpi_env
    "ASAN_OPTIONS=fast_unwind_on_malloc=0"
    "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan.supp:fast_unwind_on_malloc=0")
endif()

# ---- Serial unit tests (cstfem_core only)
foreach(t test_bc test_case test_csr test_element test_kirsch test_mesh_reader test_patch
          test_pcg test_vtu)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE cstfem_core)
  target_compile_definitions(${t} PRIVATE
    CSTFEM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    CSTFEM_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}")
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# ---- MPI smoke test
add_executable(test_mpi_smoke test_mpi_smoke.cpp)
target_link_libraries(test_mpi_smoke PRIVATE MPI::MPI_CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_mpi_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()
foreach(n 1 2 4)
  add_test(NAME mpi_smoke_np${n}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${_mpi_extra}
            $<TARGET_FILE:test_mpi_smoke> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(mpi_smoke_np${n} PROPERTIES ENVIRONMENT "${_mpi_env}")
endforeach()

# ---- Partitioning, distribution, ownership (MPI)
add_executable(test_partition test_partition.cpp)
target_link_libraries(test_partition PRIVATE cstfem)
target_compile_definitions(test_partition PRIVATE
  CSTFEM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
foreach(n 1 2 3 4)
  add_test(NAME test_partition_np${n}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${_mpi_extra}
            $<TARGET_FILE:test_partition> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(test_partition_np${n} PROPERTIES ENVIRONMENT "${_mpi_env}")
endforeach()

# ---- End-to-end Kirsch runs of cstfem_solve on the committed L0 and L1 meshes
set(_e2e_dirs "${CMAKE_CURRENT_BINARY_DIR}/e2e_kirsch_L0" "${CMAKE_CURRENT_BINARY_DIR}/e2e_kirsch_L1")
# Remove old outputs first so the checks only see files written by this run.
add_test(NAME e2e_clean COMMAND ${CMAKE_COMMAND} -E rm -rf ${_e2e_dirs})
set_tests_properties(e2e_clean PROPERTIES FIXTURES_SETUP e2e_clean)
foreach(level 0 1)
  set(dir "${CMAKE_CURRENT_BINARY_DIR}/e2e_kirsch_L${level}")
  set(case "${CMAKE_CURRENT_BINARY_DIR}/kirsch_L${level}_e2e.case")
  file(WRITE "${case}"
"mesh             = ${CMAKE_CURRENT_SOURCE_DIR}/data/kirsch_L${level}.msh
plane            = plane_stress
E                = 210e9
nu               = 0.3
thickness        = 0.01
rtol             = 1e-10
max_iter         = 20000
output_dir       = ${dir}
bc.dirichlet     = left_sym:ux=0 ; bottom_sym:uy=0
bc.traction      = kirsch_exact
kirsch.sigma_inf = 100e6
kirsch.a         = 0.1
")
  add_test(NAME e2e_kirsch_L${level}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${_mpi_extra}
            $<TARGET_FILE:cstfem_solve> "${case}" ${MPIEXEC_POSTFLAGS})
  set_tests_properties(e2e_kirsch_L${level} PROPERTIES
    FIXTURES_SETUP e2e_kirsch FIXTURES_REQUIRED e2e_clean ENVIRONMENT "${_mpi_env}")
endforeach()

add_test(NAME e2e_kirsch_check
  COMMAND ${CMAKE_COMMAND} "-DDIRS=${_e2e_dirs}" -DLIMIT=1e-9
          -P "${CMAKE_CURRENT_SOURCE_DIR}/check_e2e.cmake")
set_tests_properties(e2e_kirsch_check PROPERTIES FIXTURES_REQUIRED e2e_kirsch)

# More than one rank must be rejected clearly until Milestone 4.
add_test(NAME e2e_reject_np2
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${_mpi_extra}
          $<TARGET_FILE:cstfem_solve> "${CMAKE_CURRENT_BINARY_DIR}/kirsch_L0_e2e.case"
          ${MPIEXEC_POSTFLAGS})
set_tests_properties(e2e_reject_np2 PROPERTIES
  PASS_REGULAR_EXPRESSION "parallel path arrives in Milestone 4" ENVIRONMENT "${_mpi_env}")

# Optional: decode the VTU with Python (needs numpy; skipped if unavailable).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                  RESULT_VARIABLE _numpy_rc OUTPUT_QUIET ERROR_QUIET)
  if(_numpy_rc EQUAL 0)
    foreach(dir IN LISTS _e2e_dirs)
      get_filename_component(_name "${dir}" NAME)
      add_test(NAME ${_name}_check_vtu
        COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/python/check_vtu.py" "${dir}")
      set_tests_properties(${_name}_check_vtu PROPERTIES FIXTURES_REQUIRED e2e_kirsch)
    endforeach()
  else()
    message(STATUS "numpy not found for ${Python3_EXECUTABLE}: VTU Python check not registered")
  endif()
endif()
