#!/bin/bash 
#SBATCH --job-name=memprof
#SBATCH --output=%x.o%j
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=02:59:00
#SBATCH --cpus-per-task=32

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
module load cuda/11.8.0/gcc-11.2.0 gcc/11.2.0/gcc-4.8.5
. ~/singularity-env.sh

echo "Running NCU profiler"

CSV_OUT="${LOG_PATH}/${OUT_FILENAME}.csv"

#will be sent to singularity exec command
export COMMAND="ncu \
--kernel-name-base mangled \
--kernel-name regex:AdvX \
--metrics launch__shared_mem_per_block_dynamic,sm__warps_active.avg.per_cycle_active,launch__registers_per_thread,launch__grid_size,launch__block_size \
--target-processes all \
--csv \
--log-file $CSV_OUT \
$EXECUTABLE $ARGS"

singularity exec \
--env OMP_NUM_THREADS=36 \
--env OMP_PROC_BIND=true \
--env OMP_PLACES=cores \
--env TMPDIR=/gpfs/users/millana/TMPDIR_CONTAINERS \
--nv \
$CONTAINERSDIR/sycl-complete_latest.sif $COMMAND


echo "Profiling OK. Out file: $CSV_OUT"
