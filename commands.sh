#1 +120
#main

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt-get install -y cuda-toolkit-13-0
# echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc
# bash install_miniconda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# make CUDAPATH=/usr/local/cuda-13.0
# ./gpu_burn 36000000000

# screen -ls
# nvidia-smi

# pkill -f gpu_burn 2>/dev/null || true
nvidia-smi

conda create -n vlm python=3.11 -y
conda activate vlm

cd VLM_Distillation-main

#rm -rf ./vlm
#python -m venv vlm 
#source vlm/bin/activate
pip install -r requirements.txt

bash download_datatrain.sh

export CUDA_VISIBLE_DEVICES=4,5

bash run_baseline.sh

#9
