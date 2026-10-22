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

# ---- Partitioning/distribution and distributed solver (MPI), on 1-4 ranks
foreach(t test_partition test_distributed)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE cstfem)
  target_compile_definitions(${t} PRIVATE CSTFEM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  foreach(n 1 2 3 4)
    add_test(NAME ${t}_np${n}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${_mpi_extra}
              $<TARGET_FILE:${t}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(${t}_np${n} PROPERTIES ENVIRONMENT "${_mpi_env}")
  endforeach()
  # Same tests with ParMETIS (only in builds with -DUSE_PARMETIS=ON; not in CI).
  if(USE_PARMETIS)
    foreach(n 2 3 4)
      add_test(NAME ${t}_parmetis_np${n}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${_mpi_extra}
                $<TARGET_FILE:${t}> --partitioner=parmetis ${MPIEXEC_POSTFLAGS})
      set_tests_properties(${t}_parmetis_np${n} PROPERTIES ENVIRONMENT "${_mpi_env}")
    endforeach()
  endif()
endforeach()

# ---- End-to-end Kirsch runs of cstfem_solve on the committed L0 and L1 meshes
set(_e2e_dirs "${CMAKE_CURRENT_BINARY_DIR}/e2e_kirsch_L0" "${CMAKE_CURRENT_BINARY_DIR}/e2e_kirsch_L1")
# Parallel-output runs on L1 (rtol 1e-12): serial reference, distributed path on 1 rank, 2, 4.
# Each entry: name|path|ranks
set(_par_runs "serial_np1|auto|1" "distributed_np1|distributed|1" "np2|auto|2" "np4|auto|4")
set(_par_dirs "")
foreach(run IN LISTS _par_runs)
  string(REPLACE "|" ";" _r "${run}")
  list(GET _r 0 name)
  list(APPEND _par_dirs "${CMAKE_CURRENT_BINARY_DIR}/e2e_par_${name}")
endforeach()
# Remove old outputs first so the checks only see files written by this run.
add_test(NAME e2e_clean COMMAND ${CMAKE_COMMAND} -E rm -rf ${_e2e_dirs} ${_par_dirs})
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

foreach(run IN LISTS _par_runs)
  string(REPLACE "|" ";" _r "${run}")
  list(GET _r 0 name)
  list(GET _r 1 path)
  list(GET _r 2 np)
  set(dir "${CMAKE_CURRENT_BINARY_DIR}/e2e_par_${name}")
  set(case "${CMAKE_CURRENT_BINARY_DIR}/kirsch_L1_par_${name}.case")
  file(WRITE "${case}"
"mesh             = ${CMAKE_CURRENT_SOURCE_DIR}/data/kirsch_L1.msh
plane            = plane_stress
E                = 210e9
nu               = 0.3
thickness        = 0.01
path             = ${path}
rtol             = 1e-12
max_iter         = 20000
output_dir       = ${dir}
bc.dirichlet     = left_sym:ux=0 ; bottom_sym:uy=0
bc.traction      = kirsch_exact
kirsch.sigma_inf = 100e6
kirsch.a         = 0.1
")
  add_test(NAME e2e_par_${name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_mpi_extra}
            $<TARGET_FILE:cstfem_solve> "${case}" ${MPIEXEC_POSTFLAGS})
  set_tests_properties(e2e_par_${name} PROPERTIES
    FIXTURES_SETUP e2e_par FIXTURES_REQUIRED e2e_clean ENVIRONMENT "${_mpi_env}"
    FAIL_REGULAR_EXPRESSION "error:|WARNING|NOT converged")
  set(expect_path "${path}")
  if(path STREQUAL "auto")
    if(np EQUAL 1)
      set(expect_path serial)
    else()
      set(expect_path distributed)
    endif()
  endif()
  add_test(NAME e2e_par_${name}_timings
    COMMAND ${CMAKE_COMMAND} "-DDIR=${dir}" -DEXPECT_PATH=${expect_path} -DEXPECT_RANKS=${np}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/check_timings.cmake")
  set_tests_properties(e2e_par_${name}_timings PROPERTIES FIXTURES_REQUIRED e2e_par)
endforeach()

# path = serial with more than one rank is a clear error.
add_test(NAME e2e_path_serial_np2_rejected
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${_mpi_extra}
          $<TARGET_FILE:cstfem_solve> "${CMAKE_CURRENT_BINARY_DIR}/kirsch_L1_serial.case"
          ${MPIEXEC_POSTFLAGS})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/kirsch_L1_serial.case"
"mesh = ${CMAKE_CURRENT_SOURCE_DIR}/data/kirsch_L1.msh
plane = plane_stress
E = 210e9
nu = 0.3
thickness = 0.01
path = serial
")
set_tests_properties(e2e_path_serial_np2_rejected PROPERTIES
  PASS_REGULAR_EXPRESSION "path = serial needs exactly 1 rank" ENVIRONMENT "${_mpi_env}")

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
    # Parallel output vs the serial run: displacement by Gmsh tag, element arrays by element
    # id, gathered CSVs; relative max difference < 1e-8.
    foreach(name distributed_np1 np2 np4)
      add_test(NAME e2e_par_${name}_vs_serial
        COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/python/check_vtu.py"
                "${CMAKE_CURRENT_BINARY_DIR}/e2e_par_${name}"
                --compare "${CMAKE_CURRENT_BINARY_DIR}/e2e_par_serial_np1" --tol 1e-8)
      set_tests_properties(e2e_par_${name}_vs_serial PROPERTIES FIXTURES_REQUIRED e2e_par)
    endforeach()
  else()
    message(STATUS "numpy not found for ${Python3_EXECUTABLE}: VTU Python check not registered")
  endif()
endif()
