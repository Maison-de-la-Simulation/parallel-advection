#gcc:11 has git, gcc, python, unzip already installed
FROM gcc:11

LABEL org.opencontainers.image.source=https://github.com/maison-de-la-simulation/parallel-advection

COPY spack.yaml /opt/spack.yaml

ENV SPACK_SOURCE=spack/share/spack/setup-env.sh

WORKDIR /opt

RUN git clone https://github.com/spack/spack.git
RUN . $SPACK_SOURCE \
&& spack env create advection-env spack.yaml \
&& spack env activate advection-env \
&& spack compiler find \
&& spack concretize
RUN . $SPACK_SOURCE && spack env activate advection-env \
&& spack fetch
RUN . $SPACK_SOURCE && spack env activate advection-env \
&& spack install
# RUN . $SPACK_SOURCE && spack env activate advection-env \
# && git clone https://github.com/OpenSYCL/OpenSYCL.git --branch=v0.9.4 \
# && cd OpenSYCL/ \
# && mkdir build && cd build \
# && spack load llvm \
# # && cd $LLVM_PATH/../llvm-1* \
# # && export LLVM_PATH=$(pwd) \
# # && export HIPSYCL_INSTALL=/opt/install/hipsycl \
# && cd /opt/OpenSYCL/build \
# && cmake .. -DCMAKE_INSTALL_PREFIX=$HIPSYCL_INSTALL -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=$BOOST_ROOT -DROCM_PATH=$ROCM_PATH -DWITH_ACCELERATED_CPU=OFF \
# && make -j 8 \
# && make install \
# && export PATH=$PATH:$HIPSYCL_INSTALL/bin/ \
# && cd /opt \
# && git clone https://github.com/Maison-de-la-Simulation/parallel-advection.git \
# && cd parallel-advection && mkdir build && cd build \
# && export SYCLCC=syclcc \
# && export HIPSYCL_TARGETS='omp;hip:gfx90a' \
# && cmake .. -DCMAKE_CXX_COMPILER=$SYCLCC \
# && make -j 4