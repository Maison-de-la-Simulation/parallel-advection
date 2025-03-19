module purge

module load gcc/12.2.0
module load cmake/3.23.2
module load rocm/6.3.2
module load cuda/12.3.0
module load intel_compute_runtime/release/821.36

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/soft/compilers/cuda/cuda-12.3.0/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ac.amillan/install/intel-llvm/lib
export PATH=$PATH:/home/ac.amillan/install/intel-llvm/bin
# export OCL_ICD_FILENAMES=libigdrcl.so:libintelocl.so
