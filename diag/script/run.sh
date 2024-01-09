#!/bin/bash 

if [ "$#" != "5" ]; then
  echo "you must provide the paths to the executable (+ arguments), where to \
store log files and number of runs."
  echo "usage : ./run EXECUTABLE ARGS LOG_PATH NB_RUNS PREFIX"
  exit
fi

EXECUTABLE=$1
ARGS=$2
LOG_PATH=$3
NB_RUNS=$4
PREFIX=$5

echo "Running ${NB_RUNS} times"

for i in $(seq 0 $NB_RUNS)
do
  LOG_FILE=${LOG_PATH}/run${PREFIX}_${i}.log

   #if i is 0, we don't time this so we don't output logs
  if [[ "$i" == "0" ]]
  then
    $EXECUTABLE $ARGS
    # singularity exec --nv $CONTAINERSDIR/sycl-complete_latest.sif $EXECUTABLE $ARGS
  else
    $EXECUTABLE $ARGS > $LOG_FILE
    # singularity exec --nv $CONTAINERSDIR/sycl-complete_latest.sif $EXECUTABLE $ARGS > $LOG_FILE
  fi
done

echo "${NB_RUNS} runs OK."
