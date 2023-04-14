#!/bin/bash 
#SBATCH --job-name=adv1dsycl
#SBATCH --output=%x.o%j
#SBATCH --time=00:59:00
#SBATCH --nodes=1
##BATCH --exclusive
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

#main program to run
EXECUTABLE=~/source/advection/build/src/advection
NB_RUNS=2
#arguments for the main program
INI_FILE=~/advectionBase.ini

#path to save tmp logs
LOG_PATH=~/source/advection/benchmark/log
#name of the outputted csv file
OUT_FILENAME=perf

# load modules here
module load llvm/13.0.0/gcc-11.2.0 cuda/11.5.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0 gcc/11.2.0/gcc-4.8.5

# call run script with input parameters
./script/run $EXECUTABLE $INI_FILE $LOG_PATH $NB_RUNS

# call parse scripts with log_path and out_filename parameters
./script/parse $LOG_PATH $OUT_FILENAME

#run python plot
module load pyhton