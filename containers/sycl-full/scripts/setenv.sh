export DPCPP_ROOT=/opt/compilers/dpcpp
export DPCPP_BIN=$DPCPP_ROOT/bin
export DPCPP_LIB=$DPCPP_ROOT/lib

export ACPP_ROOT=/opt/compilers/acpp
export ACPP_BIN=$ACPP_ROOT/bin
export ACPP_LIB=$ACPP_ROOT/lib

export DPCPP=$DPCPP_BIN/clang++
export ACPP=$ACPP_BIN/syclcc

export ACPP_TARGETS="omp" #so we can compile out of the box
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DPCPP_LIB