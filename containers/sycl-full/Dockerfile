FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt upgrade -y

RUN apt install -y curl wget build-essential git sudo
RUN apt install -y python3 python3-venv python3-dev python3-pip
RUN apt install -y lsb-release software-properties-common gnupg
RUN apt install -y cmake ninja-build
RUN apt install -y libboost-all-dev

WORKDIR /opt

RUN pip install matplotlib numpy pandas

############################
#       Install LLVM       #
############################
RUN wget --progress=bar:force https://apt.llvm.org/llvm.sh \
&& chmod +x llvm.sh \
&& sudo ./llvm.sh 15 \
&& sudo apt install -y libclang-15-dev clang-tools-15 libomp-15-dev \
&& rm llvm.sh

############################
#       Install CUDA       #
############################
RUN mkdir -p /opt/cuda \
&& wget --progress=bar:force -O cuda.sh https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run \
&& sudo sh ./cuda.sh --override --silent --toolkit --no-man-page --no-drm --no-opengl-libs --installpath=/opt/cuda || true \
&& echo "CUDA Version 11.8.0" | sudo tee /opt/cuda/version.txt \
&& rm cuda.sh

############################
#       Install ROCm       #
############################
RUN sudo apt install -y libnuma-dev cmake unzip \
&& wget --progress=bar:force -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - \
&& echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.5.1 focal main" | sudo tee /etc/apt/sources.list.d/rocm.list \
&& printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600 \
&& sudo apt update -y \
&& sudo apt install -y rocm-dev

############################
#     Install LevelZero    #
############################
RUN wget https://github.com/oneapi-src/level-zero/releases/download/v1.2.3/level-zero-devel_1.2.3+u18.04_amd64.deb \
&& wget https://github.com/oneapi-src/level-zero/releases/download/v1.2.3/level-zero_1.2.3+u18.04_amd64.deb \
&& sudo dpkg -i ./level-zero* \
&& rm level-zero-devel_1.2.3+u18.04_amd64.deb \
&& rm level-zero_1.2.3+u18.04_amd64.deb

RUN mkdir source
RUN mkdir compilers

############################
#    DPC++ (requirement)   #
############################
RUN sudo apt update \
&& sudo apt install -y ca-certificates gpg wget \
&& wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
&& echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null \
&& sudo apt update \
&& sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg \
&& sudo apt install -y kitware-archive-keyring

#esimd requirement
RUN sudo apt install -y libva-dev libffi-dev libtool

#OpenCL requirement
RUN sudo apt install -y clinfo intel-opencl-icd opencl-headers

#check clinfo
RUN clinfo



############################
#           DPC++          #
############################
WORKDIR /opt/source

RUN git clone https://github.com/intel/llvm -b 2023-WW27
 
WORKDIR /opt/source/llvm

#Update cmake to > 3.25
RUN sudo apt-get update
RUN sudo apt-get install -y cmake

# configure dpcpp
RUN CUDA_LIB_PATH=/opt/cuda/lib64/stubs \
    python3 buildbot/configure.py \
    --llvm-external-projects compiler-rt \
        --cuda \
        --hip \
        --enable-plugin native_cpu \
        --cmake-opt="-DCMAKE_INSTALL_PREFIX=/opt/compilers/dpcpp" \
        --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda" \
        --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/rocm"

RUN CUDA_LIB_PATH=/opt/cuda/lib64/stubs \
    python3 buildbot/compile.py

# # WORKDIR /opt/source/llvm/build

# # RUN ninja all tools/libdevice/libsycldevice
# # RUN ninja install

# WORKDIR /home/docker/compilers_gits
RUN rm -r /opt/source/llvm

############################
#        AaaptiveCpp       #
############################
WORKDIR /opt/source

RUN git clone --recurse-submodules https://github.com/AdaptiveCpp/AdaptiveCpp.git \
&& cd AdaptiveCpp && git checkout aa00030be7ace53ea5716c6237e210a56e2227de

WORKDIR /opt/source/AdaptiveCpp

RUN mkdir build && cd build

WORKDIR /opt/source/AdaptiveCpp/build
RUN cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
        -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-15 \
        -DLLVM_DIR=/usr/lib/llvm-15/cmake \
        -DWITH_CUDA_BACKEND=ON -DWITH_ROCM_BACKEND=ON \
        -DWITH_LEVEL_ZERO_BACKEND=ON \
        -DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda \
        -DROCM_PATH=/opt/rocm \
        -DCMAKE_INSTALL_PREFIX=/opt/compilers/acpp \
        ..

RUN ninja install
WORKDIR /opt
RUN rm -rf source


############################
#         Advection        #
############################
RUN git clone https://github.com/Maison-de-la-Simulation/parallel-advection.git -b containers

LABEL org.opencontainers.image.source=https://github.com/maison-de-la-simulation/parallel-advection

COPY scripts/setenv.sh /opt/setenv.sh
