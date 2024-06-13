#!/bin/bash

if [ "$#" != "3" ]; then
  echo "you must provide the paths to the run log files and the output filename"
  echo "usage : ./parse LOG_PATH OUT_FILENAME PREFIX"
  exit
fi

LOG_PATH=$1
OUT_FILENAME=$2
PREFIX=$3

FILES="${LOG_PATH}/run${PREFIX}_*"
CSV_OUT="${LOG_PATH}/${OUT_FILENAME}.csv"

echo "Parsing outputs..."

echo "duration;cellspersec;globalsize;nx;nvx;maxIter;error;throughput;gpu" > $CSV_OUT

for f in $FILES
do
  duration=$(grep elapsed_time $f | cut -d ' ' -f2)
  cellspersec=$(grep upd_cells_per_sec $f | cut -d ' ' -f2)

  globalsize=$(grep parsing $f | cut -d ';' -f2)
  nx=$(grep parsing $f | cut -d ';' -f3)
  nvx=$(grep parsing $f | cut -d ';' -f4)
  maxIter=$(grep maxIter $f | cut -d ':' -f2 | sed 's/^ *//g')

  error=$(grep cumulated $f | cut -d ':' -f2 |  sed 's/^ *//g')
  throughput=$(grep estimated_throughput $f | cut -d ':' -f2 |  cut -d ' ' -f2)
  gpu=$(grep gpu $f | cut -d ':' -f2 | sed 's/^ *//g')

  echo "\
${duration};\
${cellspersec};\
${globalsize};\
${nx};\
${nvx};\
${maxIter};\
${error};\
${throughput};\
${gpu}" >> $CSV_OUT

done

echo "Parsing OK."
rm $FILES
