#!/bin/bash 
#SBATCH --job-name=memprof
#SBATCH --output=%x.o%j
#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH --account=cin4492
#SBATCH --constraint=MI250
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128

#main program to run
EXECUTABLE=$1

#path to save tmp logs
LOG_PATH=$2

#name of the outputted csv file
OUT_FILENAME=$3

#arguments for the main program
INI_FILE=$4

#prefix to append before logfiles
PREFIX=$5


# load modules here
module load rocm/5.3.0
export SINGULARITY_BIND="/lus/home/CT6/cin4492/amilan:/mnt,/proc,/sys,/dev,/lus/work/CT6/cin4492/amilan/ADVECTION_LOGS"

echo "Running ROCPROF profiler"

CSV_OUT="${LOG_PATH}/${OUT_FILENAME}.csv"

#will be sent to singularity exec command
export COMMAND="rocprof \
-i INPUT_ROCPROF.txt \
-o $CSV_OUT \
$EXECUTABLE $INI_FILE"

singularity exec \
--env OMP_NUM_THREADS=64 \
--env OMP_PROC_BIND=spread \
--env OMP_PLACES=cores \
--rocm \
$CONTAINERSDIR/sycl-complete_llvm-16.sif $COMMAND


echo "Profiling OK. Out file: $CSV_OUT"
