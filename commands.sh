#1 +10
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

# conda create -n python311 python=3.11 -y
conda activate python311

cd Rebuttal_VLM2Vec_Matryoshka

rm -rf ./vlm
python -m venv vlm 
source vlm/bin/activate
pip install -r requirements.txt
python download.py

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
python fix_lib.py
bash script_train/fastvlm_adaptive_mrl_laplacian_only_stage1_vqa.sh
bash script_train/fastvlm_adaptive_mrl_projection_only_stage1_vqa.sh
bash script_train/qwen3vl_adaptive_mrl_laplacian_only_stage1_cls.sh
bash script_train/qwen3vl_adaptive_mrl_projection_only_stage1_cls.sh
bash script_train/qwen3vl_adaptive_mrl_laplacian_only_stage1_vqa.sh
bash script_train/qwen3vl_adaptive_mrl_projection_only_stage1_vqa.sh
bash eval.sh
bash eval2.sh
bash eval5.sh
bash eval6.sh
bash eval_vqa.sh
bash eval_vqa2.sh
bash eval_vqa3.sh
bash eval_vqa4.sh

#9
