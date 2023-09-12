FROM intel/oneapi

COPY spack.yaml ~/spack.yaml

ENV SPACK_SOURCE=spack/share/spack/setup-env.sh

WORKDIR /opt

RUN apt update && apt install cmake -y \
&& cd /opt \
&& git clone https://github.com/Maison-de-la-Simulation/parallel-advection.git \
&& cd parallel-advection && mkdir build && cd build \
&& cmake .. -DCMAKE_CXX_COMPILER=dpcpp \
&& make -j 4

# RUN . ~/$SPACK_SOURCE && spack env activate advection-env \
# && git clone https://github.com/Maison-de-la-Simulation/parallel-advection.git \
# && cd parallel-advection && mkdir build && cd build \
# && export HIPSYCL_TARGETS='omp;cuda:sm_80' \
# && cmake .. -DCMAKE_CXX_COMPILER=syclcc \
# && make -j 8
