#!/bin/bash

module load gcc/11.2.0/gcc-4.8.5 cmake/3.21.4/gcc-11.2.0

export HIPSYCL_TARGETS="cuda-nvcxx:cc80"

export CXX=/gpfs/workdir/millana/install/opensycl-nvhpc23.1-clang16/bin/syclcc
