cmake_minimum_required(VERSION 3.16)
project(mk_cpu LANGUAGES CXX)

# ---------------- Options ----------------
option(MK_BUILD_PYTHON "Build Python extension" ON)
option(MK_USE_OPENMP   "Enable OpenMP" ON)
option(MK_NATIVE_OPT   "Enable -march=native" ON)
option(MK_USE_AVX512   "Enable AVX-512 kernels" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------- Dependencies ----------------
if (MK_USE_OPENMP)
  find_package(OpenMP)
endif()
add_subdirectory(third_party/pybind11)

# ---------------- Library ----------------
add_library(mk STATIC
  src/gemm.cpp
  src/autotuner.cpp
)
target_include_directories(mk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(mk PRIVATE ${CMAKE_SOURCE_DIR}/third_party/xbyak)

# Compile flags — pick ONE of your two bundles (this one favors raw speed)
target_compile_options(mk PRIVATE
  -O3 -Ofast -ffast-math -funroll-loops -frename-registers -DNDEBUG
)
if (MK_NATIVE_OPT)
  target_compile_options(mk PRIVATE -march=native)
endif()

if (MK_USE_AVX512)
  target_compile_definitions(mk PRIVATE MK_USE_AVX512=1)
  if (NOT MSVC)
    target_compile_options(mk PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx512vl)
  endif()
endif()

if (MK_USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(mk PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(mk PUBLIC MK_USE_OPENMP=1)
endif()

# ---------------- Benchmark executable ----------------
add_executable(mk_bench src/main.cpp)
target_link_libraries(mk_bench PRIVATE mk)
target_compile_options(mk_bench PRIVATE -O3 -Ofast -ffast-math -funroll-loops -frename-registers)
if (MK_NATIVE_OPT)
  target_compile_options(mk_bench PRIVATE -march=native)
endif()
if (MK_USE_AVX512 AND NOT MSVC)
  target_compile_options(mk_bench PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx512vl)
endif()
set_target_properties(mk_bench PROPERTIES OUTPUT_NAME "mk")

# ---------------- Python module ----------------
if (MK_BUILD_PYTHON)
  pybind11_add_module(mk_cpu src/bindings.cpp)
  target_link_libraries(mk_cpu PRIVATE mk)
  set_target_properties(mk_cpu PROPERTIES
    OUTPUT_NAME "_mk_cpu"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/mk_cpu"
  )
endif()
