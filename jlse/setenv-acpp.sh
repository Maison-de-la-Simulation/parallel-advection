module purge

module load gcc/12.2.0
module load cmake/3.23.2
module load cuda/12.3.0
module load rocm/6.3.2
module load intel_compute_runtime/release/821.36

. ~/spack/share/spack/setup-env.sh
spack env activate llvm
spack load llvm

export CMAKE_PREFIX_PATH=/home/ac.amillan/install/acpp-full
export ACPP_TARGETS=generic
export PATH=$PATH:/home/ac.amillan/install/acpp-full/bin
export benchmark_DIR=/home/ac.amillan/source/parallel-advection/thirdparty/benchmark/build

echo "Please export ACPP_VISIBILITY_MASK=cuda;hip;omp depending on the vendor"
# export ACPP_VISIBILITY_MASK=cuda;hip;omp
