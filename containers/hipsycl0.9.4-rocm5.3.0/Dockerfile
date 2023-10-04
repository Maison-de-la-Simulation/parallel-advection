FROM rocm/dev-ubuntu-22.04:5.3-complete

LABEL org.opencontainers.image.source=https://github.com/maison-de-la-simulation/parallel-advection

COPY spack.yaml /opt/spack.yaml

ENV SPACK_SOURCE=/opt/spack/share/spack/setup-env.sh

WORKDIR /opt

RUN apt update && apt install git -y \
&& git clone https://github.com/OpenSYCL/OpenSYCL.git --branch=v0.9.4 \
&& git clone https://github.com/Maison-de-la-Simulation/parallel-advection.git \
&& git clone https://github.com/spack/spack.git && cd spack \
&& git checkout 9d00bcb2864ee5a4c841f2178435aaec9f25a808  && cd /opt \
# RUN . $SPACK_SOURCE \
&& . $SPACK_SOURCE \
&& spack env create advection-env spack.yaml \
&& spack env activate advection-env \
&& spack compiler find \
&& spack concretize \
# RUN . $SPACK_SOURCE && spack env activate advection-env \
&& spack fetch
RUN . $SPACK_SOURCE && spack env activate advection-env \
&& spack install \
# RUN . $SPACK_SOURCE && spack env activate advection-env \
&& cd /opt/OpenSYCL/ \
&& mkdir build && cd build \
&& export HIPSYCL_INSTALL=/opt/install/hipsycl \
&& cmake .. \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_INSTALL_PREFIX=$HIPSYCL_INSTALL \
-DCMAKE_BUILD_TYPE=Release \
-DROCM_PATH=/opt/rocm \
-DWITH_ROCM_BACKEND=ON \
&& make -j `nproc` \
&& make install \
# RUN . $SPACK_SOURCE && spack env activate advection-env \
&& export PATH=$PATH:/opt/install/hipsycl/bin \
&& cd /opt/parallel-advection && mkdir build && cd build \
&& export HIPSYCL_TARGETS='omp;hip:gfx90a' \
&& cmake .. -DCMAKE_CXX_COMPILER=syclcc \
&& make -j `nproc` \
# RUN . $SPACK_SOURCE && spack env activate advection-env \
&& cp /opt/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.2.0/llvm-15.0.0-msh3zo5a3gnd6uig4w3cugte4a4nrnx5/lib/libomp.so /usr/lib \
&& spack uninstall -y llvm@15

CMD . $SPACK_SOURCE && spack env activate advection-env \
&& /opt/parallel-advection/build/src/advection