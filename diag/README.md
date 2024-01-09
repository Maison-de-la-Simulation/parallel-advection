# Benchmark scripts
Submit a script to the scheduler executing advection program N times and stores the results in a csv file.

## How to
Use the `orchestrate.py` script to run and parse the output of advection runs:
1. Set correct parameters in `bench_ini.py`
2. Set correct modules in `launch.sh`
3. Set cluster parameters in  `launch.sh` header
4. If using containers, update command in `script/run.sh` with container command (e.g. `singularity exec ...`)
