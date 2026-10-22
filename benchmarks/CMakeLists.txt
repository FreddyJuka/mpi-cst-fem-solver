# Micro-benchmarks (not part of the solver; not run by CTest).
add_executable(stream_triad stream_triad.cpp)
target_link_libraries(stream_triad PRIVATE MPI::MPI_CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(stream_triad PRIVATE -Wall -Wextra -Wpedantic)
endif()
