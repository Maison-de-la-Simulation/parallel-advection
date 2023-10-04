FROM gcc:11

LABEL org.opencontainers.image.source=https://github.com/maison-de-la-simulation/parallel-advection

ENV BIN_ADVECTION=/opt/parallel-advection/build/src/advection

# wget https://github.com/oneapi-src/level-zero/releases/download/v1.2.3/level-zero-devel_1.2.3+u18.04_amd64.deb
COPY level-zero-devel_1.2.3+u18.04_amd64.deb /opt/level-zero-devel_1.2.3+u18.04_amd64.deb
# wget https://github.com/oneapi-src/level-zero/releases/download/v1.2.3/level-zero_1.2.3+u18.04_amd64.deb
COPY level-zero_1.2.3+u18.04_amd64.deb /opt/level-zero_1.2.3+u18.04_amd64.deb
# wget -q -O cuda.sh https://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
COPY cuda.sh /opt/cuda.sh


WORKDIR /opt

RUN apt update \
&& apt install -y ocl-icd-opencl-dev \
&& apt install -y libtbb-dev \
&& apt install -y libnuma-dev \
&& apt install -y cmake unzip git
# RUN apt install -y libboost-all-dev

RUN dpkg -i ./level-zero*
RUN mkdir -p /opt/OpenSYCL/cuda
# && wget -q -O cuda.sh https://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
RUN chmod +x cuda.sh && ./cuda.sh --toolkit --no-man-page --no-drm --no-opengl-libs --installpath=/opt/OpenSYCL/cuda \
&& nvcc --version | sudo tee /opt/OpenSYCL/cuda/version.txt
# RUN 
#     - name: install ROCm
#         [[ ${{matrix.rocm}} == 4.0.1 ]] && CODENAME=xenial
#         [[ ${{matrix.rocm}} == 5.4.3 ]] && CODENAME=focal
#         sudo 
#         wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
#         echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${{matrix.rocm}} $CODENAME main" | sudo tee /etc/apt/sources.list.d/rocm.list
#         printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600
#         sudo apt update
#         sudo apt install rocm-dev
#     - name: install LLVM
#         wget https://apt.llvm.org/llvm.sh
#         chmod +x llvm.sh
#         sudo ./llvm.sh ${{matrix.clang}}
#         sudo apt install libclang-${{matrix.clang}}-dev clang-tools-${{matrix.clang}} libomp-${{matrix.clang}}-dev
#         if [[ "${{matrix.clang}}" == "16" ]]; then
#           sudo rm -r /usr/lib/clang/16*
#           sudo ln -s /usr/lib/llvm-16/lib/clang/16 /usr/lib/clang/16
#         fi
#     - name: install boost (from apt)
#       run: |
#         sudo apt install 
#     - name: build Open SYCL
#       run: |
#         mkdir build && cd build
#         cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++-${{matrix.clang}} -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-${{matrix.clang}} -DLLVM_DIR=/usr/lib/llvm-${{matrix.clang}}/cmake -DWITH_CUDA_BACKEND=ON -DWITH_ROCM_BACKEND=ON -DWITH_LEVEL_ZERO_BACKEND=ON -DCMAKE_INSTALL_PREFIX=`pwd`/install -DCUDA_TOOLKIT_ROOT_DIR=/opt/OpenSYCL/cuda -DROCM_PATH=/opt/rocm ..
#         make -j2 install
#         cp /opt/OpenSYCL/cuda/lib64/stubs/libcuda.so `pwd`/install/lib/libcuda.so
#         cp /opt/OpenSYCL/cuda/lib64/stubs/libcuda.so `pwd`/install/lib/libcuda.so.1

# RUN rm *.deb && rm cuda.sh