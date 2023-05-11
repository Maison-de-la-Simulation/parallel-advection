#!/bin/bash
module load cuda/11.7.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0 gcc/11.2.0/gcc-4.8.5
export HIPSYCL_TARGETS='omp;cuda:sm_80'
export CXX=/gpfs/workdir/millana/install/opensycl-clang16-cuda11.7/bin/syclcc

