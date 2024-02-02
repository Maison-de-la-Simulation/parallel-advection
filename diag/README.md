# Benchmark scripts
Submit a script to the scheduler executing advection program N times and stores the results in a csv file.

## How to: Singularity benchmarks `launch_benchmarks.sh`
1. Update the `launch_benchmarks.sh` script with:
   1. The right SLURM configuration
   2. The path towards the build directory and output logs directory
   3. The filters for benchmarks (e.g. `BM_Advector/0` to benchmark advector for CPU, e.g. `BM_WgSize/1` to benchmark work-group sizes for GPU, ...)
2. Submit to scheduler with e.g. `sbatch launch_benchmarks.sh`


## How to: `orchestrate.py`
Use the `orchestrate.py` script to run and parse the output of advection runs:
1. Set correct parameters in `bench_ini.py`
2. Set correct modules in `launch.sh`
3. Set cluster parameters in  `launch.sh` header
4. If using containers, update command in `script/run.sh` with container command (e.g. `singularity exec ...`)

