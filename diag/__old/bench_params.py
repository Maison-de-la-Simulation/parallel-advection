import os.path as path

# Root directory of the code for host and container
HOST_ROOTDIR = "/gpfs/users/millana/source/parallel-advection"
CONTAINER_ROOTDIR = "/mnt/parallel-advection"  # or "" if no container

# FULL path to save tmp logs (on the host if RUN or PARSE)
# inside container if PROFILE mode
LOG_PATH = "/gpfs/workdir/millana/ADVECTION_LOGS"

# RELATIVE path to the executable
EXECUTABLE = "build/src/advection"

# RELATIVE path to direcroty containing the .ini file
INIFILE_DIR = "benchmark/script/ncu-ini"
# INIFILE_DIR = "benchmark/script"

# FULL path to the main output file with mean, std, and all infos for all runs
GLOBAL_CSV_FILE = path.join(HOST_ROOTDIR,
                            "benchmark/log/describe_all_ruche_cpu.csv")

SETS = {
    "kernelImpl": ["BasicRange2D",
                   "BasicRange1D",
                   "Hierarchical",
                   "Scoped",
                   "NDRange"],
    "use_gpu": [True,
                False],
    "(n1,nvx)": [(1024, 2**x) for x in range(8, 20)],
}

## PROFILE MODE OPT
# IMPL="DPCPP"
IMPL="ACPP"