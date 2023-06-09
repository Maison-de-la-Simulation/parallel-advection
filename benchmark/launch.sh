#!/bin/bash 
#SBATCH --job-name=adv1dsycl
#SBATCH --output=%x.o%j
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:59:00
#SBATCH --partition=cpu_med

#####SBATCH --gres=gpu:1
#####SBATCH --partition=gpua100

NB_RUNS=5

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
module load cuda/11.7.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0 gcc/11.2.0/gcc-4.8.5

# call run script with input parameters
./script/run $EXECUTABLE $INI_FILE $LOG_PATH $NB_RUNS $PREFIX
rm $INI_FILE

# call parse scripts with log_path and out_filename parameters
./script/parse $LOG_PATH $OUT_FILENAME $PREFIX