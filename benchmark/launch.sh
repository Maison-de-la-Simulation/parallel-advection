#!/bin/bash 
#SBATCH --job-name=adv1dsycl
#SBATCH --output=%x.o%j
#SBATCH --time=00:59:00
#SBATCH --nodes=1
##BATCH --exclusive
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

EXECUTABLE=~/source/advection/build/src/advection
INI_FILE=~/advectionBase.ini
LOG_PATH=~/source/advection/benchmark/log
NB_RUNS=2

# load modules here
module load llvm/13.0.0/gcc-11.2.0 cuda/11.5.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0 gcc/11.2.0/gcc-4.8.5

# call run script with input parameters
./script/run $EXECUTABLE $INI_FILE $LOG_PATH $NB_RUNS

# call parse scripts with log_path parameter

# call python script
# ...