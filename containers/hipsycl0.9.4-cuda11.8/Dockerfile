FROM gcc:11

LABEL org.opencontainers.image.source=https://github.com/maison-de-la-simulation/parallel-advection

COPY spack.yaml /opt/spack.yaml

ENV SPACK_SOURCE=/opt/spack/share/spack/setup-env.sh
ENV BIN_ADVECTION=/opt/parallel-advection/build/src/advection

WORKDIR /opt

RUN git clone https://github.com/spack/spack.git
RUN . $SPACK_SOURCE \
&& spack env create advection-env spack.yaml \
&& spack env activate advection-env \
&& spack compiler find \
&& spack concretize \
&& spack fetch

RUN . $SPACK_SOURCE && spack env activate advection-env \
&& spack install
RUN . $SPACK_SOURCE && spack env activate advection-env \
&& git clone https://github.com/Maison-de-la-Simulation/parallel-advection.git \
&& cd parallel-advection && mkdir build && cd build \
&& export SYCLCC=syclcc \
&& export HIPSYCL_TARGETS='omp;cuda:sm_80' \
&& cmake .. -DCMAKE_CXX_COMPILER=$SYCLCC \
&& make -j 4

# Remove unused filed to empty storage

# ENTRYPOINT [".", "/opt/spack/share/spack/setup-env.sh && \"$@\"", "-s"]
# ENTRYPOINT ["spack env activate", "advection-env && \"$@\"", "-s"]
CMD $BIN_ADVECTION