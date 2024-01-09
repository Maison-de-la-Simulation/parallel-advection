#!/bin/bash 
#SBATCH --job-name=adv1dsycl
#SBATCH --output=%x.o%j
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=01:59:00


NB_RUNS=10

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
./script/run.sh $EXECUTABLE $INI_FILE $LOG_PATH $NB_RUNS $PREFIX

# call parse scripts with log_path and out_filename parameters
./script/parse.sh $LOG_PATH $OUT_FILENAME $PREFIX
