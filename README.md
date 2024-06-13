# Parallel Advection

This code implements a 1D advection operator inside a multidimensionnal space. It implements a [semi-Lagrangian scheme](https://en.wikipedia.org/wiki/Semi-Lagrangian_scheme) using the [SYCL 2020](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) progamming models.

To reproduce the benchmark, follow the [benchmark README.md](benchmark/README.md) instructions.

## General algorithm
For one time step, this is the algorithm's structure:

![Advection process](docs/fig/AdvectionProcess.png)

### SYCL Implementations
We tried to directly map the batched problem onto 4 different SYCL parallel loops (BasicRange, Hierarchical, NDRange, Scoped)
Impls (folder `src/core/impl`):


# Build the project (see instructions on branches):
You can use the `compile.sh` script to compile for various hardware and sycl-implementations.
Help command `compile.sh --help`:

```sh
Simple compilation script. Automatically builds the project for a combination (hw, sycl).
For multiple devices compilation flows, please compile manually.
Usage: ./compile.sh [--hw <mi250|a100|x86_64>] [--sycl <intel-llvm|acpp|oneapi>] [--benchmark_DIR=<directory>]
Compilers must be present in PATH:
           intel-llvm : clang++
           acpp       : syclcc
           oneapi     : icpx
```

Example usage: `./compile.sh --hw a100 --sycl intel-llvm --benchmark_DIR=/path/to/google/benchmark/build` will generate the `build_intel-llvm_a100` folder with the executables.

## Manually build the project
Flags depends on the SYCL implementation you are using.
- For DPC++, add the correct flags via the `-DDPCPP_FSYCL_TARGETS` cmake variable.
- For acpp, export the `ACPP_TARGETS` environment variable before compiling
```sh
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/path/to/sycl/compiler
make
```

# Run the executable
1. Set the runtime parameters in `build/src/advection.ini`

```ini
[run]
# Total number of iterations
maxIter = 200
# Wheter to run on the GPU or CPU #only for SYCL
gpu     = true
# The kernel type to use for Xadvection
kernelImpl  = Hierarchical  
# Size of work groups use in the kernels
workGroupSize = 32
# Outputs a solution.log file to be read with the python notebook
outputSolution = false

[geometry]
nx  = 1024 # nb of spatial points
nvx = 2 # nb of speed points
n_fict_dim = 1  # nb of points in the fictive dimension

[discretization]
dt  = 0.001

minRealX  = 0
maxRealX  = 1
minRealVx = -1
maxRealVx = 1
```

Deltas $d_{vx}$ and $d_x$ are deduced by the number of points and min/max values.

1. Run the executable `build/src/advection`


### Credits
This code is largely inspired by the [vlp4D](https://github.com/yasahi-hpc/vlp4d) code
