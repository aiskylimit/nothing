#1 +30
#sql-main
#v3

#2 -f-/home/ubuntu/aiskylimit_nothing/text2sql_distillation_draft/run_logs/20260718_160742/jobs/ +a
# nvidia-smi
# ls /home/ubuntu/aiskylimit_nothing/text2sql_distillation_draft/results/eval/synid_ce_multilayer_3/qwen/spider_data/seed42

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt-get install -y cuda-toolkit-13-0
# echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc
# bash install_miniconda.sh

# cd gpu_burn
# make CUDAPATH=/usr/local/cuda-13.0
# ./gpu_burn 36000000000

# pkill -f gpu_burn 2>/dev/null || true

kill -9 $(pgrep -f "/.venv/bin/python")
sleep 20
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# cd text2sql_distillation_draft
# source .venv/bin/activate
# python -c "import nltk; nltk.download('punkt_tab')"
# bash scripts/qwen/synid_ce_multilayer_3/format_eval_multiseed.sh
# bash ./project_commands.sh
