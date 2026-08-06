#2
#dpo
#v1

#2 -f-/home/ubuntu/aiskylimit_nothing/segd_distillation/logs/ +a
# nvidia-smi

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt-get install -y cuda-toolkit-13-0
# sudo apt install -y zip unzip
# echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc
# bash install_miniconda.sh

# cd gpu_burn
# make CUDAPATH=/usr/local/cuda-13.0
# ./gpu_burn 36000000000

# kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
# sleep 5
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH



cd Gen-DPO
rm -rf .venv-tis-dpo
CUDA_VISIBLE_DEVICES=2,3 bash ./project_commands.sh

# cd ./talas_vlm_embed
# ls
# ls training
# ls training/FastVLM-0.5B_talas_1.0_eos_norm_proj_cls
# bash ./project_commands.sh

# cd ./segd_distillation
# bash ./project_commands.sh
