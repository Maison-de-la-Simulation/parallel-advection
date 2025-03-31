#!/bin/bash

export HW=pvc
export SYCL=acpp
export APP=conv1d
export REP=10

export OUTFILE="/home/ac.amillan/source/parallel-advection/jlse/out/$APP/$HW/$SYCL.json"
export EXEC="/home/ac.amillan/source/parallel-advection/build_${SYCL}_${HW}/benchmark/$APP-bench"

# --benchmark_report_aggregates_only=true \
$EXEC --benchmark_counters_tabular=true \
      --benchmark_min_warmup_time=1 \
      --benchmark_format=json \
      --benchmark_repetitions=$REP \
      --benchmark_out=$OUTFILE
