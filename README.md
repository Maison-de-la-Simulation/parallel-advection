# Parallel Advection

This code implements a 1D advection operator inside a multidimensionnal space. It implements a [semi-Lagrangian scheme](https://en.wikipedia.org/wiki/Semi-Lagrangian_scheme) using C++ parallel progamming models (SYCL, Kokkos).

## General algorithm
![Advection process](docs/fig/AdvectionProcess.png)

### SYCL Implementations
**Tools:**
- `local_accessor`: to allocate shared memory between work items of a same work group
- groups barriers on work groups
- simple mapping between physical and logical iteration space

Impls (folder `sycl/src/core/.../impl`):
- Basic Range
  - 1D
  - 2D
- ND-Range
- Hierarchical
- Scoped

### Kokkos Implementations
**Tools:**
- `ScratchSpace` memory
- Barriers

Implementations (folder `kokkos/src/core/.../impl`):
- MDRange
- TeamPolicy


# Build the project (see instructions on branches):
```sh
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/path/to/sycl/compiler
make
```

### Credits
This code is largely inspired by the [vlp4D](https://github.com/yasahi-hpc/vlp4d) code