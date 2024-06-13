# Running benchmarks
This section is part of the reproductibility initiative of SC24.
Benchmark are implemented using the [google benchmark](https://github.com/google/benchmark/) microbenchmark suite.

## Prerequisites
- Apptainer/Singularity must be installed on the machine.
- Pull the [sycl-complete container](https://github.com/Maison-de-la-Simulation/parallel-advection/pkgs/container/sycl-complete).
  - `apptainer pull docker://ghcr.io/maison-de-la-simulation/sycl-container:llvm-16`
- Build [google benchmark](https://github.com/google/benchmark) library with classical compiler (can be built in the submodule folder `thirdparty/benchmark`).
- A cluster with SLURM scheduler

## Instructions
1) Inside the container: compile the `parallel-advection` project with benchmark option (see [README](../README.md)). The file `main_bench.cpp` contains the main bencmarks.

2) Update the `launch_benchmark.sh` script on a SLURM scheduled cluster to submit the benchmarks to the compute nodes
   - Select the right SLURM environment in header
   - Update `CONTAINER_RUN` (full path to the `.sif` file)
   - Update `OUTFILE_DIR` (directory where to output the .json file with benchmark resutls) and `BUILD_DIR` (directory containing the benchmark executable)
   - Update correct values for `HW` (hardware), `IMPL` (sycl implementation), `ARGS_GPU_SINGULARITY` if GPU benchmarking (typically `--nv` or `--rocm`)
   - Use the `BENCHMARK_FILTER` environment variable. e.g. `BENCHMARK_FILTER=BM_Advector/0` or `BM_Advector/1` (0 for CPU 1 for GPU)

3) Once the `launch_benchmark.sh` script is updated with correct values, submit it to the SLURM scheduler with `sbatch launch_benchmarks.sh`

## Full example: A100 GPU with AdaptiveCpp
```sh
# Inside parallel-advection folder with submodules pulled

$ export APPTAINER_BIND="`pwd`:/par-adv"
$ apptainer pull docker://ghcr.io/maison-de-la-simulation/sycl-container:llvm-16
$ apptainer shell sycl-container_llvm-16.sif

#inside container
Singularity> cd /par-adv/thirdparty/benchmark
Singularity> mkdir build && cmake .. -D.... #compile google benchmark apart
Singularity> cd /par-adv
Singularity> ./compile.sh --hw a100 --sycl acpp --benchmark_BUILD_DIR=/par-adv/thirdparty/benchmark/build
Singularity> exit #get out of the container

#On a cluster with SLURM,
#Update values for SLURM scheduler and correct values for benchmark (here we target a100 and acpp, we will need the --nv flag for singularity)
$ vi diag/launch_benchmarks.sh
$ sbatch diag/launch_benchmarks.sh #run the benchmarks for A100
```

# Visualize results
To visualize the results with python (pandas and matplotlib are required):
```python
from utils.py import * #import diag/utils.py functions to vizualise
df_acpp_a100 = get_cleaned_df("LOGS/A100/a100_acpp_50rep.json") #turn json into DataFrame
a100_acpp = create_dict_from_df(df_acpp_a100) #turn DataFrame into a easily plotable list
plot_values(a100_acpp, "A100 GPU ACPP", do_show=true) #plot the values and show
```

# Known issues
- With AdaptiveCpp on the MI250X, export the `ACPP_PERSISTENT_RUNTIME=1` variable because it causes crash with google benchmark framework.