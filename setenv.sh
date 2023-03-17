#!/bin/bash
module load llvm/13.0.0/gcc-11.2.0 cuda/11.5.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0 gcc/11.2.0/gcc-4.8.5
export HIPSYCL_TARGETS='cuda:sm_80'
export CXX=/gpfs/users/millana/source/hipSYCL/build/install/bin/syclcc

