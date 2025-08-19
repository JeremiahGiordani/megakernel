cmake_minimum_required(VERSION 3.16)
project(attn_cpu LANGUAGES CXX)

# ---------------- Options ----------------
option(ATTN_BUILD_PYTHON "Build Python extension" ON)
option(ATTN_USE_OPENMP   "Enable OpenMP" ON)
option(ATTN_NATIVE_OPT   "Enable -march=native" ON)
option(ATTN_USE_AVX512   "Enable AVX-512 kernels" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------- Dependencies ----------------
if (ATTN_USE_OPENMP)
  find_package(OpenMP)
endif()
add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/xbyak)

# ---------------- Library ----------------
add_library(attn STATIC
  src/attn_dispatch.cpp
  src/gemm.cpp
  src/jit_gemm.cpp
)
target_include_directories(attn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(attn PRIVATE ${CMAKE_SOURCE_DIR}/third_party/xbyak)

# Compile flags — pick ONE of your two bundles (this one favors raw speed)
target_compile_options(attn PRIVATE
  -O3 -Ofast -ffast-math -funroll-loops -frename-registers -DNDEBUG
)
if (ATTN_NATIVE_OPT)
  target_compile_options(attn PRIVATE -march=native)
endif()

if (ATTN_USE_AVX512)
  target_compile_definitions(attn PRIVATE ATTN_USE_AVX512=1)
  if (NOT MSVC)
    target_compile_options(attn PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx512vl)
  endif()
endif()

if (ATTN_USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(attn PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(attn PUBLIC ATTN_USE_OPENMP=1)
endif()

# ---------------- Benchmark executable ----------------
add_executable(attn_bench src/main.cpp)
target_link_libraries(attn_bench PRIVATE attn)
target_compile_options(attn_bench PRIVATE -O3 -Ofast -ffast-math -funroll-loops -frename-registers)
if (ATTN_NATIVE_OPT)
  target_compile_options(attn_bench PRIVATE -march=native)
endif()
if (ATTN_USE_AVX512 AND NOT MSVC)
  target_compile_options(attn_bench PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx512vl)
endif()
set_target_properties(attn_bench PROPERTIES OUTPUT_NAME "attn")

# ---------------- Python module ----------------
if (ATTN_BUILD_PYTHON)
  pybind11_add_module(attn_cpu src/bindings.cpp)
  target_link_libraries(attn_cpu PRIVATE attn)
  set_target_properties(attn_cpu PROPERTIES
    OUTPUT_NAME "_attn_cpu"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/attn_cpu"
  )
endif()
