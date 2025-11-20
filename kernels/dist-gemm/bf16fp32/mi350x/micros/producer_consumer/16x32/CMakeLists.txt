cmake_minimum_required(VERSION 3.21)
project(tk_kernel LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Download CPM.cmake
set(CPM_DOWNLOAD_VERSION 0.38.7)
if(CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
         https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
         ${CPM_DOWNLOAD_LOCATION}
    )
endif()

include(${CPM_DOWNLOAD_LOCATION})

# Disable iris examples, benchmarks, and tests to speed up build
set(IRIS_BUILD_EXAMPLES OFF CACHE BOOL "Build iris examples" FORCE)
set(IRIS_BUILD_BENCHMARKS OFF CACHE BOOL "Build iris benchmarks" FORCE)
set(IRIS_BUILD_TESTS OFF CACHE BOOL "Build iris tests" FORCE)

# Ensure spdlog is built with -fPIC for shared library linking
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add iris library from GitHub (it will auto-fetch its dependencies like spdlog)
CPMAddPackage(
    NAME iris
    GITHUB_REPOSITORY ROCm/iris
    GIT_TAG muhaawd/irisx
    SOURCE_SUBDIR irisx
    OPTIONS
        "IRIS_BUILD_EXAMPLES OFF"
        "IRIS_BUILD_BENCHMARKS OFF"
        "IRIS_BUILD_TESTS OFF"
)

# GPU Target
set(GPU_TARGET "CDNA4" CACHE STRING "GPU target architecture")
set(SRC "micro_02_2stage_8c4p.cpp")

# Auto-detect THUNDERKITTENS_ROOT (go up from current directory to project root)
if(NOT DEFINED THUNDERKITTENS_ROOT)
    set(THUNDERKITTENS_ROOT $ENV{THUNDERKITTENS_ROOT})
endif()

if(NOT THUNDERKITTENS_ROOT)
    # Auto-detect: go up 7 levels from kernels/dist-gemm/bf16fp32/mi350x/micros/producer_consumer/16x32
    get_filename_component(THUNDERKITTENS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../" ABSOLUTE)
    message(STATUS "Auto-detected THUNDERKITTENS_ROOT: ${THUNDERKITTENS_ROOT}")
endif()

# ROCm/HIP setup
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

if(NOT ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
endif()

message(STATUS "Using ROCM_PATH: ${ROCM_PATH}")
message(STATUS "Using THUNDERKITTENS_ROOT: ${THUNDERKITTENS_ROOT}")

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")

# Set GPU-specific flags
if(GPU_TARGET STREQUAL "CDNA2")
    set(GPU_FLAGS "-DKITTENS_CDNA2 --offload-arch=gfx90a")
elseif(GPU_TARGET STREQUAL "CDNA3")
    set(GPU_FLAGS "-DKITTENS_CDNA3 --offload-arch=gfx942")
elseif(GPU_TARGET STREQUAL "CDNA4")
    set(GPU_FLAGS "-DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math")
endif()

# MPI setup
find_package(MPI REQUIRED)

# Python setup
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# Create the tk_kernel module
pybind11_add_module(tk_kernel ${SRC})

# Set compile options
target_compile_options(tk_kernel PRIVATE
    ${GPU_FLAGS}
    -w
    -Rpass-analysis=kernel-resource-usage
)

# Include directories
target_include_directories(tk_kernel PRIVATE
    ${THUNDERKITTENS_ROOT}/include
    ${THUNDERKITTENS_ROOT}/prototype
    ${ROCM_PATH}/include/hip
)

# Link libraries (iris::iris will bring in its own includes and dependencies)
target_link_libraries(tk_kernel PRIVATE
    iris::iris
    ${Python3_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    -ldl
    -lm
)

# Set output name and directory to match Python extension naming
set_target_properties(tk_kernel PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tk_kernel"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Create the iris_py module
pybind11_add_module(iris_py iris_py.cpp)

# Set compile options for iris_py
target_compile_options(iris_py PRIVATE
    -w
)

# Link libraries for iris_py
target_link_libraries(iris_py PRIVATE
    iris::iris
    ${Python3_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    -ldl
    -lm
)

# Set output name and directory to match Python extension naming
set_target_properties(iris_py PROPERTIES
    PREFIX ""
    OUTPUT_NAME "iris_py"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Custom target for logging
add_custom_command(TARGET tk_kernel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build completed: tk_kernel.${Python3_SOABI}.so"
)

add_custom_command(TARGET iris_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build completed: iris_py.${Python3_SOABI}.so"
)

