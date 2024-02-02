#!/bin/bash

# =================================================
# compile.sh
# =================================================
HARDWARE=""
SYCL_IMPL=""
BENCHMARK_DIR=""

ONEAPI_COMPILER="icpx"
INTELLLVM_COMPILER="clang++"
ACPP_COMPILER="syclcc"

usage() {
    echo "Simple compilation script. Automatically builds the project for a combination (hw, sycl)."
    echo "For multiple devices compilation flows, please compile manually."
    echo "Usage: $0 [--hw <mi250|a100|x86_64>] [--sycl <intel-llvm|acpp|oneapi>] [--benchmark_DIR=<directory>]"
    echo "Compilers must be present in PATH:"
    echo "           intel-llvm : ${INTELLLVM_COMPILER}"
    echo "           acpp       : ${ACPP_COMPILER}"
    echo "           oneapi     : ${ONEAPI_COMPILER}"
    exit 1
}

# =================================================
# Argument parsing
# =================================================
while [ "$#" -gt 0 ]; do
    case $1 in
        --hw)
            HARDWARE="$2"
            shift 2
            ;;
        --sycl)
            SYCL_IMPL="$2"
            shift 2
            ;;
        --benchmark_DIR=*)
            BENCHMARK_DIR="${1#*=}"
            shift 1
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$HARDWARE" ] || [ -z "$SYCL_IMPL" ]; then
    echo "Error: Both --hw and --sycl options are required."
    usage
fi

ERR_SYCL_UNKNOWN="### Error: Unsupported SYCL implementation specified."
ERR_HW_UNKNOWN="### Error: Unsupported hardware architecture specified."

# =================================================
# Set right CXX compiler (must be in PATH)
# =================================================
if [ "$SYCL_IMPL" == "intel-llvm" ]; then
    CMAKE_OPTIONS+=" -DCMAKE_CXX_COMPILER=${INTELLLVM_COMPILER}"
elif [ "$SYCL_IMPL" == "acpp" ]; then
    CMAKE_OPTIONS+=" -DCMAKE_CXX_COMPILER=${ACPP_COMPILER}"
elif [ "$SYCL_IMPL" == "oneapi" ]; then
    CMAKE_OPTIONS+=" -DCMAKE_CXX_COMPILER=${ONEAPI_COMPILER}"
else
    echo $ERR_SYCL_UNKNOWN
    usage
fi

# =================================================
# Set CMake options based on hardware and compiler
# =================================================
if [ "$HARDWARE" == "mi250" ]; then
    if [ "$SYCL_IMPL" == "intel-llvm" ]; then
        CMAKE_OPTIONS+=" -DDPCPP_FSYCL_TARGETS='-fsycl-targets=amd_gpu_gfx90a'"
    elif [ "$SYCL_IMPL" == "acpp" ]; then
        export ACPP_TARGETS="hip:gfx90a"
    elif [ "$SYCL_IMPL" == "oneapi" ]; then
        echo "### Warning: Combination not tested: ${SYCL_IMPL} - ${HARDWARE}"
    else
        echo $ERR_SYCL_UNKNOWN
        usage
    fi
elif [ "$HARDWARE" == "a100" ]; then
    # Add options for a100 and different SYCL implementations
    if [ "$SYCL_IMPL" == "intel-llvm" ]; then
        CMAKE_OPTIONS+=" -DDPCPP_FSYCL_TARGETS='-fsycl-targets=nvidia_gpu_sm_80'"
    elif [ "$SYCL_IMPL" == "acpp" ]; then
        export ACPP_TARGETS="cuda:sm_80"
    elif [ "$SYCL_IMPL" == "oneapi" ]; then
        echo "### Warning: Combination not tested: ${SYCL_IMPL} - ${HARDWARE}"
    else
        echo $ERR_SYCL_UNKNOWN
        usage
    fi
elif [ "$HARDWARE" == "x86_64" ]; then
    # Add options for x86_64 and different SYCL implementations
    if [ "$SYCL_IMPL" == "intel-llvm" ]; then
        CMAKE_OPTIONS+=" -DDPCPP_FSYCL_TARGETS='-fsycl-targets=spir64_x86_64'"
    elif [ "$SYCL_IMPL" == "acpp" ]; then
        export ACPP_TARGETS="omp"
    # elif [ "$SYCL_IMPL" == "oneapi" ]; then
        #do nothing
    else
        echo $ERR_SYCL_UNKNOWN
        usage
    fi
else
    echo $ERR_HW_UNKNOWN
    usage
fi

# Add benchmark directory option if specified
if [ -n "$BENCHMARK_DIR" ]; then
    CMAKE_OPTIONS+=" -Dbenchmark_DIR=${BENCHMARK_DIR}"
fi

# =================================================
# Configure
# =================================================
# =================================================
# Configure
# =================================================
BUILD_DIR=build_${SYCL_IMPL}_${HARDWARE}

# Check if the build directory exists
if [ -d "${BUILD_DIR}" ]; then
    echo "### Removing existing build directory: ${BUILD_DIR}"
    rm -r "${BUILD_DIR}"
fi

mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

echo "### Configuring project..."
cmake $CMAKE_OPTIONS ..

# Check the exit status of the CMake configuration
if [ $? -ne 0 ]; then
    echo "### Error: CMake configuration failed. Deleting the build directory."
    rm -r "$BUILD_DIR"
    exit 1
fi
echo "### CMake configuration complete in `pwd`."
echo ""

# =================================================
# Build
# =================================================
echo "### Building project..."
cmake --build . --parallel 24

# Check the exit status of the build
if [ $? -ne 0 ]; then
    echo "### Error: Build failed."
    exit 1
fi

echo "### Build complete in `pwd`."
echo ""
