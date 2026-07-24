#2 -f-/home/ubuntu/aiskylimit_nothing/text2sql_distillation_draft/run_logs/llama_synid_sql/20260724_075447/jobs/ +a
#sql-main
#v1

#2 -f-/home/ubuntu/aiskylimit_nothing/text2sql_distillation_draft/run_logs/20260723_150003/jobs/ +a
#2 -f-/home/ubuntu/aiskylimit_nothing/text2sql_distillation_draft/results/eval/synid_ce_keywords_weight_lora_436/qwen_updated/collect/ +a
# nvidia-smi

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

# kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

# sleep 20
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


cd text2sql_distillation_draft
mv results results_ablation
bash ./project_commands_llama.sh
# ls text2sql_distillation_draft/results -R
# source .venv/bin/activate
# RUN_GPUS=0,1,2,3,4,5,6,7 bash scripts/qwen_ablation_3/infer_csd_ckpt1090.sh
# RUN_GPUS=0,1,2,3,4,5,6,7 CKPT_STEP=874 bash scripts/qwen_ablation_3/infer_csd_ckpt1090.sh
cd ./results
zip -r eval_infer.zip eval infer
du -sh eval_infer.zip
# bash ./eval.sh
# python -c "import nltk; nltk.download('punkt_tab')"
# bash scripts/qwen/synid_ce_multilayer_3/format_eval_multiseed.sh

# cd ark
# bash ./project_commands.sh
