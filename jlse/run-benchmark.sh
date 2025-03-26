#!/bin/bash

export OUTFILE="/home/ac.amillan/source/parallel-advection/jlse/out/pvc/acpp.json"
export REP=10
# export BENCHMARK_FILTER=BM_Advector/0

export EXEC="/home/ac.amillan/source/parallel-advection/build_dpcpp_pvc/benchmark/paper-bench"

$EXEC --benchmark_counters_tabular=true \
      --benchmark_report_aggregates_only=true \
      --benchmark_min_warmup_time=1 \
      --benchmark_format=json \
      --benchmark_repetitions=$REP \
      --benchmark_out=$OUTFILE
