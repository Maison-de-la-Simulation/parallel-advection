# Running benchmarks
This section is part of the reproductibility initiative of SC24.
## Prerequisites
- Apptainer/Singularity or Docker must be installed on the machine.
- Pull the [sycl-complete container](https://github.com/Maison-de-la-Simulation/parallel-advection/pkgs/container/sycl-complete).
  - `apptainer pull docker://ghcr.io/maison-de-la-simulation/sycl-container:llvm-16`
- Build [google benchmark](https://github.com/google/benchmark) library with classical compiler (can be built in the submodule folder `thirdparty/benchmark`).
- A cluster with SLURM scheduler

## Instructions
1) Inside the container: compile the `parallel-advection` project with `-Dbenchmark_DIR` option, this will build the benchmark files, `main_bench.cpp` contains the main bencmarks.

2) Update the `launch_benchmark.sh` script on a SLURM scheduled cluster to submit the benchmarks to the compute nodes
   - Select the right SLURM environment in header
   - Update `CONTAINER_RUN` (full path to the `.sif` file)
   - Update `OUTFILE_DIR` (directory where to output the .json file with benchmark resutls)and `BUILD_DIR` (directory containing the benchmark executable)
   - Update correct values for `HW` (hardware), `IMPL` (sycl implementation), `ARGS_GPU_SINGULARITY` if GPU benchmarking (typically `--nv` or `--rocm`)
   - Use the `BENCHMARK_FILTER` environment variable. e.g. `BENCHMARK_FILTER=BM_Advector/0` or `BM_Advector/1` (0 for CPU 1 for GPU)

3) Once the `launch_benchmark.sh` script is updated with correct values, submit it to the SLURM scheduler with `sbatch launch_benchmarks.sh`

## Full example: A100 GPU with AdaptiveCpp
```sh
# Once container is pulled

apptainer pull docker://ghcr.io/maison-de-la-simulation/sycl-container:llvm-16
# Inside parallel-advection folder with submodules pulled
export APPTAINER_BIND="`pwd`:/par-adv"
apptainer shell sycl-container_llvm-16.sif
#inside container
cd /par-adv/thirdparty/benchmark
mkdir build && cmake .. -D.... #compile google benchmark apart
cd /par-adv
./compile.sh --hw a100 --sycl acpp --benchmark_DIR /par-adv/thirdparty/benchmark/build
#main executable is build_a100_acpp/src/advection
exit #get out of the container

#On a cluster with SLURM,
vi diag/launch_benchmarks.sh #Update values for SLURM scheduler and correct values for benchmark (here we target a100 and acpp, we will need the --nv flag for singularity)
sbatch launch_benchmarks.sh #run the benchmarks for A100
```

# Visualize results
To visualize the results with python (pandas and matplotlib are required):
```python
from utils.py import * #import diag/utils.py functions to vizualise
df_acpp_a100 = get_cleaned_df("LOGS/A100/a100_acpp_50rep.json") #turn json into DataFrame
a100_acpp = create_dict_from_df(df_acpp_a100) #turn DataFrame into a easily plotable list
plot_values(a100_acpp, "A100 GPU ACPP", do_show=true) #plot the values and show
```
