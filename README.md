# Batched Advection kernels

This code implements a 1D advection operator inside a multidimensionnal space. It implements a [semi-Lagrangian scheme](https://en.wikipedia.org/wiki/Semi-Lagrangian_scheme) using the [SYCL 2020](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) progamming models.

To reproduce the benchmark, follow the [benchmark README.md](benchmark/README.md) instructions.

## General algorithm
For one time step, the algorithm's structure is as follow:

![Advection process](docs/fig/AdvectionProcess.png)

### SYCL Implementations
The algorithm is implemented in various ways using different SYCL constructs. It requires local memory allocation via the [local accessor](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:accessor.local). The implementations are in the `src/core/impl` directory.

# Build the project:
You can use the `compile.sh` script to compile for various hardware and sycl-implementations. For multi-device compilation flows, build the project manually.
Use the `./compile.sh --help` to see the options.

Example usage:
```sh
#generate the advection executable and advection.ini file
./compile.sh --hw x86_64 --sycl intel-llvm 

#create build_intel-llvm_a100 folder with benchmarks
./compile.sh --hw a100 --sycl intel-llvm --benchmark_BUILD_DIR=/path/to/google/benchmark/build 

#create build_acpp_mi250 folder with tests and execute tests
./compile.sh --hw mi250 --sycl acpp --build-tests --run-tests 
```

## Manually build the project
Flags varies on the SYCL implementation you are using.
- For DPC++, add the correct flags via the `-DDPCPP_FSYCL_TARGETS` cmake variable.
- For acpp, export the `ACPP_TARGETS` environment variable before compiling

# Run the executable
1. Set the runtime parameters in `build/src/advection.ini`

    ```ini
    [run]
    # Total number of iterations
    maxIter = 100
    # Wheter to run on the GPU or CPU
    gpu     = true
    # The kernel type to use for advection
    kernelImpl  = Hierarchical
    # Size of work groups use in the kernels
    workGroupSizeX = 128
    workGroupSizeY = 1
    # Outputs a solution.log file to be read with the python notebook
    outputSolution = false

    [geometry]
    nb = 512   # nb of speed points (batch dimension)
    n1 = 1024 # nb of spatial points (dimension of interest)
    n2 = 2 # fictive dimension, is also stride for x-dim

    [discretization]
    dt  = 0.001

    minRealX  = 0
    maxRealX  = 1
    minRealVx = -1
    maxRealVx = 1
    ```

    Deltas $d_{vx}$ and $d_x$ are deduced by the number of points and min/max values.

2. Run the executable `build/src/advection`


### Credits
This code is largely inspired by the [vlp4D](https://github.com/yasahi-hpc/vlp4d) code.
