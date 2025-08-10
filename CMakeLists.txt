cmake_minimum_required(VERSION 3.20)
project(megakernel_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimize hard by default for the runtime
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MKCPU_ENABLE_ASAN "Enable AddressSanitizer" ON)
if (MKCPU_ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(DetectSIMD)

if (NOT MK_HAS_AVX512)
  message(FATAL_ERROR "AVX-512 is required for this MVP. Please build on a machine/compiler with AVX-512 support.")
endif()



# ---- Library ----
add_library(mkcpu STATIC
  src/onnx_loader.cpp
  src/regionizer.cpp
  src/weight_packing.cpp
  src/conv_microkernels_avx512.cpp
#   src/fused_conv_chain.cpp
  src/engine.cpp
)

target_include_directories(mkcpu PRIVATE third_party/onnx)
target_include_directories(mkcpu PUBLIC include)
target_compile_options(mkcpu PRIVATE ${MK_AVX512_FLAGS} -O3 -DNDEBUG -fvisibility=hidden)
target_compile_definitions(mkcpu PUBLIC MK_AVX512=1 MK_VEC=16)

# (Optional) OpenMP later:
# find_package(OpenMP)
# if (OpenMP_CXX_FOUND)
#   target_link_libraries(mkcpu PUBLIC OpenMP::OpenMP_CXX)
# endif()

# Protobuf/ONNX linkage as you add them:
find_package(Protobuf REQUIRED)
target_link_libraries(mkcpu PRIVATE protobuf::libprotobuf-lite)

# --- Generate ONNX protobuf (C++) ---
set(ONNX_PROTO_ROOT ${CMAKE_SOURCE_DIR}/third_party/onnx)
set(ONNX_PROTO_DIR  ${ONNX_PROTO_ROOT}/onnx)

# Choose the base proto(s) we need. ModelProto lives in onnx.proto.
set(ONNX_PROTO_FILES
    ${ONNX_PROTO_DIR}/onnx.proto
    # If you ever need operator schemas, you can also add:
    # ${ONNX_PROTO_DIR}/onnx-operators.proto
)

# Ensure protoc output directory exists
set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

# Generate C++ sources/headers into build/generated (will create onnx/ subdir)
protobuf_generate_cpp(
    ONNX_PROTO_SRCS ONNX_PROTO_HDRS
    ${ONNX_PROTO_FILES}
    IMPORT_DIRS    ${ONNX_PROTO_ROOT}
    PROTOC_OUT_DIR ${GENERATED_DIR}
)


add_library(onnx_protos STATIC ${ONNX_PROTO_SRCS})
target_include_directories(onnx_protos PUBLIC
    ${CMAKE_BINARY_DIR}/generated      # so "onnx/onnx.pb.h" is visible
    ${ONNX_PROTO_ROOT}                 # so "onnx/onnx_pb.h" is visible
)
target_link_libraries(onnx_protos PUBLIC protobuf::libprotobuf-lite)

# Some ONNX headers switch includes based on ONNX_ML. We use non-ML to keep it simple.
target_compile_definitions(onnx_protos PUBLIC ONNX_ML=0)
target_compile_definitions(mkcpu       PUBLIC ONNX_ML=0)


# Link our library against the generated protos and expose includes
target_link_libraries(mkcpu PRIVATE onnx_protos)
target_include_directories(mkcpu PRIVATE
    ${CMAKE_BINARY_DIR}/generated
    ${ONNX_PROTO_ROOT}
)

# ---- Python bindings ----
option(MKCPU_BUILD_PYTHON "Build Python bindings" ON)
if (MKCPU_BUILD_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if (NOT pybind11_FOUND)
    add_subdirectory(third_party/pybind11)
  endif()

  pybind11_add_module(mkcpu_py src/bindings.cpp)
  # Make the import name `mkcpu`
  set_target_properties(mkcpu_py PROPERTIES OUTPUT_NAME "mkcpu")
  target_link_libraries(mkcpu_py PRIVATE mkcpu)
  target_compile_options(mkcpu_py PRIVATE ${MK_AVX512_FLAGS} -O3 -DNDEBUG -fvisibility=hidden)
  target_compile_definitions(mkcpu_py PRIVATE MK_AVX512=1 MK_VEC=16)
  target_compile_definitions(mkcpu_py    PRIVATE ONNX_ML=0)

endif()
