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
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
endif()

# Generated header lives in the build tree.
set(CSTFEM_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(cmake/config.hpp.in "${CSTFEM_GENERATED_DIR}/cstfem/config.hpp" @ONLY)

# Serial core: mesh, case file, element, assembly, BCs, solver. No MPI, no METIS.
add_library(cstfem_core
  src/assembly.cpp
  src/bc.cpp
  src/case.cpp
  src/csr.cpp
  src/element.cpp
  src/mesh_io.cpp
  src/solver.cpp
  src/version.cpp)
target_include_directories(cstfem_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CSTFEM_GENERATED_DIR}")
target_compile_features(cstfem_core PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cstfem_core PUBLIC -Wall -Wextra -Wpedantic)
endif()

# Parallel layer: core + MPI + METIS. Interface-only until it has MPI sources (Milestone 4).
add_library(cstfem INTERFACE)
target_link_libraries(cstfem INTERFACE cstfem_core MPI::MPI_CXX METIS::METIS)
# Optional backends are exposed only through the generated config.hpp (USE_* macros).
if(USE_PARMETIS)
  target_link_libraries(cstfem INTERFACE ParMETIS::ParMETIS)
endif()
if(USE_PETSC)
  target_link_libraries(cstfem INTERFACE PkgConfig::PETSC)
endif()

add_executable(cstfem_solve src/main.cpp)
target_link_libraries(cstfem_solve PRIVATE cstfem)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
