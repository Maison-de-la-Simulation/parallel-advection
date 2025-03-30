# Batched Kernels with Memory Allocations (BKMA)

This code implements optimal parallel loops for BKMA-like algorithm. Two use case implementations are available in the code, a **1D semi-lagrangian advection operator** and a **1D Convolution operator**.

![Advection process](docs/fig/AdvectionProcess.png)

## 1D Convolution operator
Implement a [1D convolution operator](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) in-place using BKMA strategies.

## Lagrangian Advection

Implements a 1D advection operator inside a multidimensionnal space. It implements a [semi-Lagrangian scheme](https://en.wikipedia.org/wiki/Semi-Lagrangian_scheme) using the [SYCL 2020](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) progamming models.

To reproduce the benchmark, follow the [benchmark README.md](benchmark/README.md) instructions.

### SYCL Implementations
The algorithm is implemented in various ways using different SYCL constructs. It requires local memory allocation via the [local accessor](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:accessor.local). The implementations are in the `src/core` directory.

- BasicRange (out of place), no hierarchical parallelism involved
- NDRange (in-place), work-groups and work-items, direct mapping of the problem dimensions
- AdaptiveWg (in-place or out-of-place), optimized work-group sizes, streaming, optimal local memory usage

# Build the project:
You can use the `compile.sh` script to compile for various hardware and sycl-implementations. For multi-device compilation flows, build the project manually.
Use the `./compile.sh --help` to see the options.

Example usage:
```sh
#generate the advection executable and advection.ini file
./compile.sh --hw cpu --sycl dpcpp 

#create build_dpcpp_a100 folder with benchmarks
./compile.sh --hw a100 --sycl dpcpp --benchmark_DIR=/path/to/google/benchmark/build 

#create build_acpp_mi300 folder with tests and execute tests
./compile.sh --hw mi300 --sycl acpp --build-tests --run-tests 
```

## Manually build the project
Flags varies on the SYCL implementation you are using.
- For DPC++, add the correct flags via the `-DDPCPP_FSYCL_TARGETS` cmake variable.
- For acpp, export the `ACPP_TARGETS` environment variable before compiling

# Run the executable
1. Set the runtime parameters in `build/src/<conv1d|advection>.ini`
2. Run the executable `build/src/advection/<conv1d|advection>`


### Credits
The advection operator in this code is largely inspired by the [vlp4D](https://github.com/yasahi-hpc/vlp4d) code.
