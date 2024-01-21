#!/bin/bash 
#SBATCH --job-name=SYCL_Adv
#SBATCH --time=05:00:00
#SBATCH --output=%x.o%j
#SBATCH --nodes=1
#SBATCH --exclusive

#============================================================
# RUCHE
#============================================================
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpua100

###SBATCH --cpus-per-task=40
###SBATCH --partition cpu_long

#============================================================
# ADASTRA
#============================================================
# export SINGULARITY_BIND="/proc,/sys,/dev"

###SBATCH --account=cin4492
###SBATCH --gres=gpu:1
###SBATCH --gpus-per-node=1
###SBATCH --cpus-per-task=64
###SBATCH --constraint=MI250

###SBATCH --account=cin4691
###SBATCH --cpus-per-task=192
###SBATCH --constraint=GENOA

#============================================================
export CONTAINER_RUN=$CONTAINERSDIR/sycl-complete_latest.sif

export OUTFILE_DIR=/gpfs/users/my_user
export BUILD_DIR=/gpfs/users/my_user/parallel-advection/build_acpp

export HW=a100
export IMPL=acpp
export ARGS_GPU_SINGULARITY= #--rocm #--nv

#Benchmark parameters
export BENCH_EXEC=${BUILD_DIR}/benchmark/main-bench
export BENCHMARK_FILTER=BM_Advector/0
export BENCHMARK_REPETITIONS=10

export OUTFILE=$OUTFILE_DIR/${HW}_${IMPL}_${BENCHMARK_REPETITIONS}rep.json
#============================================================

#============================================================
#Modules
module purge
if [ -e "~/singularity-env.sh" ]; then
    . ~/singularity-env.sh
fi
export SINGULARITY_BIND="$SINGULARITY_BIND,$OUTFILE_DIR,$BUILD_DIR"
#============================================================

export COMMAND="$BENCH_EXEC \
--benchmark_counters_tabular=true \
--benchmark_report_aggregates_only=true \
--benchmark_min_warmup_time=1 \
--benchmark_format=json \
--benchmark_out=$OUTFILE"

singularity exec \
--env OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} \
--env OMP_PLACES=cores \
--env OMP_PROC_BIND=true \
$ARGS_GPU_SINGULARITY \
$CONTAINER_RUN $COMMAND
