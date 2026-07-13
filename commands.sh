#1 +300
#main

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt-get install -y cuda-toolkit-13-0
# bash install_miniconda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
which nvcc
lsb_release -a

# make CUDAPATH=/usr/local/cuda-13.0
# ./gpu_burn 36000000000

# screen -ls
# nvidia-smi

# pkill -f gpu_burn 2>/dev/null || true
# nvidia-smi
# git clone --branch new_ap https://github.com/DanhVinhLe/Rebuttal_VLM2Vec_Matryoshka.git
# cd Rebuttal_VLM2Vec_Matryoshka
# apt-get update
# apt-get upgrade -y
# apt install tmux zip unzip -y
# apt-get install -y libgl1 libglib2.0-0
# rm -rf ./vlm
# python -m venv vlm 
# source vlm/bin/activate
# pip install -r requirements.txt
# python download.py
# wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
# unzip images.zip -d eval_images/
# python fix_lib.py
# bash script_train/fastvlm_adaptive_mrl_laplacian_only_stage1_vqa.sh
# bash script_train/fastvlm_adaptive_mrl_projection_only_stage1_vqa.sh
# bash script_train/qwen3vl_adaptive_mrl_laplacian_only_stage1_cls.sh
# bash script_train/qwen3vl_adaptive_mrl_projection_only_stage1_cls.sh
# bash script_train/qwen3vl_adaptive_mrl_laplacian_only_stage1_vqa.sh
# bash script_train/qwen3vl_adaptive_mrl_projection_only_stage1_vqa.sh
# bash eval.sh
# bash eval2.sh
# bash eval5.sh
# bash eval6.sh
# bash eval_vqa.sh
# bash eval_vqa2.sh
# bash eval_vqa3.sh
# bash eval_vqa4.sh

#9
