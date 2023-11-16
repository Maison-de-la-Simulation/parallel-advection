# SYCL Container

This container contains an environment to run and compile [SYCL 2020](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) codes on NVIDIA and AMD GPUs and any CPU with two SYCL compilers:
- AdaptiveCpp [v23.10.0](https://github.com/AdaptiveCpp/AdaptiveCpp/tree/v23.10.0)
- oneAPI DPC++ [589824d](https://github.com/intel/llvm/tree/589824d74532c85dee50e006cdc6685269eadfef) 

## How to run

Compilers are installed in `/opt/sycl`. The environment is setup in the `$PATH` so just type in `sycl-ls` or `acpp-info`.

### Singularity
```sh
singularity pull docker://ghcr.io/maison-de-la-simulation/sycl-complete
..
sycl-ls
acpp-info
```

### Docker
```sh
docker pull ghcr.io/maison-de-la-simulation/sycl-complete
..
```

## Tested hardware
We tested with a version of docker at least ... and singularity ...

**GPUs**
| NVIDIA            |      AMD      |
|:-----------------:|:-------------:|
| A100 (`sm_80`)    |    MI250x     |
| V100              |               |

**CPUs**
|       Intel      | AMD        |
|:----------------:|:----------:|
| Xeon Gold 6230   | EPYC ...   |
| ...              | GENOA ...  |

## Details
Versions of the backend used inside the container:
- **NVIDIA GPUs:** CUDA 11.8 via [NVIDIA base container](https://hub.docker.com/layers/nvidia/cuda/11.8.0-devel-ubuntu20.04/images/sha256-6e12af425102e25d3e644ed353072eca3aa8c5f11dd79fa8e986664f9e62b37a?context=explore)
- **AMD GPUs:** [ROCm](https://docs.amd.com/en/docs-5.5.0/deploy/linux/index.html) 5.5.1
- **OpenMP CPUs** (used for AdaptiveCpp CPUs compilation and runtime): [LLVM](https://apt.llvm.org/) 16.0.0
- **OpenCL Devices** (used for DPC++ CPUs runtime): OpenCL via [oneAPI DPC++ Get Started Guide](https://intel.github.io/llvm-docs/GetStartedGuide.html#install-low-level-runtime)

Tools:
CMake 3.27, Vim

## Known issues
- clang++-16 (llvm) and clang++ (intel llvm) different
- Intel on this hash commit because `sycl-ls` get_platform segfault on CUDA with multiple backend [cf this issue](https://github.com/intel/llvm/issues/4381) and we need SYCL 2020 features such as local_accessor 
- TMPDIR on cluster for clang
- mount /sys /dev depending on singularity configuration
