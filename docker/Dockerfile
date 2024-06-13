FROM ghcr.io/maison-de-la-simulation/sycl-complete:latest

WORKDIR /

RUN git clone https://github.com/Maison-de-la-Simulation/parallel-advection.git -b 3d \
&& cd parallel-advection \
&& git submodule init \
&& git submodule update

WORKDIR /parallel-advection
