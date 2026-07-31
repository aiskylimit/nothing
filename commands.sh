#1 +120
#main
#v1

#2 -f-/home/ubuntu/aiskylimit_nothing/text2sql_distillation_draft/run_logs/20260723_150003/jobs/ +a
#2 -f-/home/ubuntu/aiskylimit_nothing/text2sql_distillation_draft/results/eval/synid_ce_keywords_weight_lora_436/qwen_updated/collect/ +a
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

kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
sleep 5
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


cd ./text2sql_distillation_draft
QWEN25_GPU_LIST=0,1,2,3,4,5,6,7 GPUS_PER_JOB=2 RUN_MODE=parallel \
BATCH_SIZE=4 GRAD_ACC=1 bash project_commands_qwen2.5.sh
