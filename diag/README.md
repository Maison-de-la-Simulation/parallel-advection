# Visualize results
To visualize the results with python (pandas and matplotlib are required):
```python
from utils.py import * #import diag/utils.py functions to vizualise
df_acpp_a100 = get_cleaned_df("LOGS/A100/a100_acpp_50rep.json") #turn json into DataFrame
a100_acpp = create_dict_from_df(df_acpp_a100) #turn DataFrame into a easily plotable list
plot_values(a100_acpp, "A100 GPU ACPP", do_show=true) #plot the values and show
```
______________________________________
______________________________________

## Benchmark scripts
### How to: Singularity benchmarks `launch_benchmarks.sh`
1. Update the `launch_benchmarks.sh` script with:
   1. The right SLURM configuration
   2. The path towards the build directory and output logs directory
   3. The filters for benchmarks (e.g. `BM_Advector/0` to benchmark advector for CPU, e.g. `BM_WgSize/1` to benchmark work-group sizes for GPU, ...)
2. Submit to scheduler with e.g. `sbatch launch_benchmarks.sh`


### How to: `orchestrate.py` (depreciated)
Use the `orchestrate.py` script to run and parse the output of advection runs:
1. Set correct parameters in `bench_ini.py`
2. Set correct modules in `launch.sh`
3. Set cluster parameters in  `launch.sh` header
4. If using containers, update command in `script/run.sh` with container command (e.g. `singularity exec ...`)

See [benchmark README](../benchmark/README.md) for details.