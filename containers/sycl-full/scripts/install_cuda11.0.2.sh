mkdir -p ~/opt/cuda
wget --progress=bar:force -O cuda.sh http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
sudo sh ./cuda.sh --override --silent --toolkit --no-man-page --no-drm --no-opengl-libs --installpath=~/opt/cuda || true
echo "CUDA Version 11.0.2" | sudo tee ~/opt/cuda/version.txt
rm cuda.sh