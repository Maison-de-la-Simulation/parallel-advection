wget --progress=bar:force https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 15
sudo apt install -y libclang-15-dev clang-tools-15 libomp-15-dev
rm llvm.sh