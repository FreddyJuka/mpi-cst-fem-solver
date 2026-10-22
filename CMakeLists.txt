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

add_library(cstfem
  src/case.cpp
  src/element.cpp
  src/mesh_io.cpp
  src/version.cpp)
target_include_directories(cstfem PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CSTFEM_GENERATED_DIR}")
target_link_libraries(cstfem PUBLIC MPI::MPI_CXX METIS::METIS)
target_compile_features(cstfem PUBLIC cxx_std_17)
# Optional backends are exposed only through the generated config.hpp (USE_* macros).
if(USE_PARMETIS)
  target_link_libraries(cstfem PUBLIC ParMETIS::ParMETIS)
endif()
if(USE_PETSC)
  target_link_libraries(cstfem PUBLIC PkgConfig::PETSC)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cstfem PUBLIC -Wall -Wextra -Wpedantic)
endif()

add_executable(cstfem_solve src/main.cpp)
target_link_libraries(cstfem_solve PRIVATE cstfem)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
