cmake_minimum_required(VERSION 3.20)
project(cstfem VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(USE_PARMETIS "Enable the ParMETIS parallel partitioner" OFF)
option(USE_PETSC "Enable the PETSc solver backend" OFF)
option(CSTFEM_MPI_OVERSUBSCRIBE "Pass --oversubscribe to mpiexec in tests (OpenMPI)" OFF)
option(CSTFEM_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(CSTFEM_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all
                      -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(METIS REQUIRED)

if(USE_PARMETIS)
  find_package(ParMETIS REQUIRED)
endif()

if(USE_PETSC)
  # PETSc via pkg-config. Besides the default search path, look in PETSC_DIR[/PETSC_ARCH]
  # (CMake or environment variable) and in the Debian/Ubuntu prefix /usr/lib/petscdir/petsc-real.
  find_package(PkgConfig REQUIRED)
  set(_petsc_dir "${PETSC_DIR}")
  if(NOT _petsc_dir AND DEFINED ENV{PETSC_DIR})
    set(_petsc_dir "$ENV{PETSC_DIR}")
  endif()
  set(_petsc_arch "${PETSC_ARCH}")
  if(NOT _petsc_arch AND DEFINED ENV{PETSC_ARCH})
    set(_petsc_arch "$ENV{PETSC_ARCH}")
  endif()
  set(_petsc_pc_dirs "")
  if(_petsc_dir)
    list(APPEND _petsc_pc_dirs "${_petsc_dir}/${_petsc_arch}/lib/pkgconfig" "${_petsc_dir}/lib/pkgconfig")
  endif()
  list(APPEND _petsc_pc_dirs "/usr/lib/petscdir/petsc-real/lib/pkgconfig")
  foreach(_d IN LISTS _petsc_pc_dirs)
    if(EXISTS "${_d}")
      set(ENV{PKG_CONFIG_PATH} "${_d}:$ENV{PKG_CONFIG_PATH}")
    endif()
  endforeach()
  pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
  message(STATUS "PETSc ${PETSC_VERSION}: ${PETSC_INCLUDE_DIRS}")

  # The backend needs PetscScalar = double (real build) and a PetscInt that holds every int
  # index (32-bit PETSc indices match ours; 64-bit ones are a safe widening).
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_LIBRARIES PkgConfig::PETSC MPI::MPI_CXX)
  check_cxx_source_compiles("
#include <petscsys.h>
#include <type_traits>
static_assert(std::is_same<PetscScalar, double>::value, \"PetscScalar\");
static_assert(sizeof(PetscInt) >= sizeof(int) && std::is_signed<PetscInt>::value, \"PetscInt\");
int main() { return 0; }" CSTFEM_PETSC_TYPES_OK)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT CSTFEM_PETSC_TYPES_OK)
    message(FATAL_ERROR "USE_PETSC=ON needs a real double-precision PETSc (PetscScalar = double) "
                        "whose PetscInt holds an int; the PETSc found does not compile the check "
                        "(see CMakeFiles/CMakeConfigureLog.yaml)")
  endif()
endif()

# Generated header lives in the build tree.
set(CSTFEM_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(cmake/config.hpp.in "${CSTFEM_GENERATED_DIR}/cstfem/config.hpp" @ONLY)

# Serial core: mesh, case file, element, assembly, BCs, solver, output, serial driver.
# No MPI, no METIS.
add_library(cstfem_core
  src/assembly.cpp
  src/bc.cpp
  src/case.cpp
  src/csr.cpp
  src/driver.cpp
  src/element.cpp
  src/kirsch.cpp
  src/mesh_io.cpp
  src/postprocess.cpp
  src/solver.cpp
  src/version.cpp
  src/vtu.cpp)
target_include_directories(cstfem_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CSTFEM_GENERATED_DIR}")
target_compile_features(cstfem_core PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cstfem_core PUBLIC -Wall -Wextra -Wpedantic)
endif()

# Parallel layer: everything that uses MPI or METIS. Never linked into cstfem_core.
add_library(cstfem
  src/distributed.cpp
  src/ghost_exchange.cpp
  src/mpi_util.cpp
  src/parallel_driver.cpp
  src/partition.cpp)
target_link_libraries(cstfem PUBLIC cstfem_core MPI::MPI_CXX METIS::METIS)
# Optional backends are exposed only through the generated config.hpp (USE_* macros).
if(USE_PARMETIS)
  target_link_libraries(cstfem PUBLIC ParMETIS::ParMETIS)
endif()
if(USE_PETSC)
  target_sources(cstfem PRIVATE src/petsc_backend.cpp)
  target_link_libraries(cstfem PUBLIC PkgConfig::PETSC)
endif()

add_executable(cstfem_solve src/main.cpp)
target_link_libraries(cstfem_solve PRIVATE cstfem)

add_subdirectory(benchmarks)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
