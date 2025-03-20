#!/bin/bash

# =================================================
# compile.sh
# =================================================
HARDWARE=""
SYCL_IMPL=""
BENCHMARK_DIR=""
RUN_TESTS=false

ONEAPI_COMPILER="icpx"
DPCPP_COMPILER="clang++"
ACPP_COMPILER="acpp"

CMAKE_OPTIONS+=" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

usage() {
    echo "Simple compilation script. Automatically builds the project for a combination (hw, sycl)."
    echo "For multiple devices compilation flows, please compile manually."
    echo "Usage: $0 [--hw <mi250|a100|cpu|mi300|pvc|h100>] [--sycl <dpcpp|acpp|oneapi>] [--benchmark_BUILD_DIR=<directory>] [--build-tests] [--run-tests] [--debug]"
    echo "Compilers must be present in PATH:"
    echo "           dpcpp      : ${DPCPP_COMPILER}"
    echo "           acpp       : ${ACPP_COMPILER}"
    echo "           oneapi     : ${ONEAPI_COMPILER}"
    exit 1
}

# =================================================
# Argument parsing
# =================================================
BUILD_TESTS=false  # Initialize the boolean variable
BUILD_DEBUG=false

while [ "$#" -gt 0 ]; do
    case $1 in
        --hw)
            HARDWARE="$2"
            shift 2  # Remove --hw and its argument from the list
            ;;
        --sycl)
            SYCL_IMPL="$2"
            shift 2  # Remove --sycl and its argument from the list
            ;;
        --benchmark_BUILD_DIR=*)
            BENCHMARK_DIR="${1#*=}"
            shift 1  # Remove --benchmark_BUILD_DIR=path from the list
            ;;
        --build-tests)
            BUILD_TESTS=true
            shift 1  # Remove --build-tests from the list
            ;;
        --run-tests)
            RUN_TESTS=true
            shift 1  # Remove --run-tests from the list
            ;;
        --debug)
            BUILD_DEBUG=true
            shift 1
            ;;
        *)
            usage  # Handle unknown options
            ;;
    esac
done


if [ -z "$HARDWARE" ] || [ -z "$SYCL_IMPL" ]; then
    echo "Error: Both --hw and --sycl options are required."
    usage
fi

if $RUN_TESTS && ! $BUILD_TESTS; then
    echo "Error: --run-tests option requires --build-tests option to be set."
    usage
fi

ERR_SYCL_UNKNOWN="### Error: Unsupported SYCL implementation specified."
ERR_HW_UNKNOWN="### Error: Unsupported hardware architecture specified."

# =================================================
# Set right CXX compiler (must be in PATH)
# =================================================
if [ "$SYCL_IMPL" == "dpcpp" ]; then
    CMAKE_OPTIONS+=" -DCMAKE_CXX_COMPILER=${DPCPP_COMPILER}"
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
    if [ "$SYCL_IMPL" == "dpcpp" ]; then
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
    if [ "$SYCL_IMPL" == "dpcpp" ]; then
        CMAKE_OPTIONS+=" -DDPCPP_FSYCL_TARGETS='-fsycl-targets=nvidia_gpu_sm_80'"
    elif [ "$SYCL_IMPL" == "acpp" ]; then
        export ACPP_TARGETS="cuda:sm_80"
    elif [ "$SYCL_IMPL" == "oneapi" ]; then
        echo "### Warning: Combination not tested: ${SYCL_IMPL} - ${HARDWARE}"
    else
        echo $ERR_SYCL_UNKNOWN
        usage
    fi
elif [ "$HARDWARE" == "cpu" ]; then
    # Add options for cpu and different SYCL implementations
    if [ "$SYCL_IMPL" == "dpcpp" ]; then
        CMAKE_OPTIONS+=" -DDPCPP_FSYCL_TARGETS='-fsycl-targets=spir64_x86_64'"
    elif [ "$SYCL_IMPL" == "acpp" ]; then
        export ACPP_TARGETS="omp"
    elif [ "$SYCL_IMPL" == "oneapi" ]; then
        # do nothing
        CMAKE_OPTIONS+=""
    else
        echo $ERR_SYCL_UNKNOWN
        usage
    fi
elif [ "$HARDWARE" == "mi300" ]; then
    if [ "$SYCL_IMPL" == "dpcpp" ]; then
        CMAKE_OPTIONS+=" -DDPCPP_FSYCL_TARGETS='-fsycl-targets=amd_gpu_gfx942'"
    elif [ "$SYCL_IMPL" == "acpp" ]; then
        export ACPP_TARGETS="generic"
    elif [ "$SYCL_IMPL" == "oneapi" ]; then
        #do nothing
        CMAKE_OPTIONS+=""
    else
        echo $ERR_SYCL_UNKNOWN
        usage
    fi
elif [ "$HARDWARE" == "pvc" ]; then
    if [ "$SYCL_IMPL" == "dpcpp" ]; then
        CMAKE_OPTIONS+=" -DBUILD_WITH_INTEL_LLVM=ON"
    elif [ "$SYCL_IMPL" == "acpp" ]; then
        export ACPP_TARGETS="generic"
    elif [ "$SYCL_IMPL" == "oneapi" ]; then
        #do nothing
        CMAKE_OPTIONS+=""
    else
        echo $ERR_SYCL_UNKNOWN
        usage
    fi
elif [ "$HARDWARE" == "h100" ]; then
    if [ "$SYCL_IMPL" == "dpcpp" ]; then
        CMAKE_OPTIONS+=" -DDPCPP_FSYCL_TARGETS='-fsycl-targets=nvidia_gpu_sm_90'"
    elif [ "$SYCL_IMPL" == "acpp" ]; then
        export ACPP_TARGETS="generic"
    elif [ "$SYCL_IMPL" == "oneapi" ]; then
        #do nothing
        CMAKE_OPTIONS+=""
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

# Add tests compilation if specified
if $BUILD_TESTS; then
    CMAKE_OPTIONS+=" -DADVECTION_BUILD_TESTS=ON"
fi

# Add tests compilation if specified
if $BUILD_DEBUG; then
    CMAKE_OPTIONS+=" -DCMAKE_BUILD_TYPE=Debug"
fi

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
echo "# CMake options: ${CMAKE_OPTIONS}"
echo ""

# =================================================
# Build
# =================================================
echo "### Building project..."
cmake --build . --parallel 16

# Check the exit status of the build
if [ $? -ne 0 ]; then
    echo "### Error: Build failed."
    exit 1
fi

echo "### Build complete in `pwd`."
echo ""

# =================================================
# Run tests if specified
# =================================================
if $RUN_TESTS; then
    echo "### Running tests..."
    ctest --output-on-failure
    echo "### Tests complete."
    echo ""
fi
