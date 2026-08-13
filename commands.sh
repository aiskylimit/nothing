#2 -0-5 +a
#chi
#v1

#2 -f-/home/ubuntu/aiskylimit_nothing/segd_distillation/logs/ +a
#2 -f-/home/ubuntu/aiskylimit_nothing/talas_vlm_embed/MMEB-evaloutputs-json/ +a
#2 -f-~/aiskylimit_nothing/gen_data/deepseek_output/split/Distill_Qwen_32B_generated_outputs_part_3.zip.part_004 +a


# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt-get install -y cuda-toolkit-13-0
# sudo apt install -y zip unzip
# echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc
# bash install_miniconda.sh

cd gpu_burn
make CUDAPATH=/usr/local/cuda-13.0
./gpu_burn 36000000000

# kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
# sleep 5
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH



# cd Gen-DPO
# source eval/.venv/bin/activate
# CUDA_VISIBLE_DEVICES=6,7 bash all.sh output/dpo_Llama-3.1-Tulu-3-8B-SFT_ultra-feedback_08-06_15-06
# CUDA_VISIBLE_DEVICES=6,7 bash all.sh output/gendpo_Llama-3.1-Tulu-3-8B-SFT_ultra-feedback_08-06_21-05
# CUDA_VISIBLE_DEVICES=6,7 bash all.sh output/sft_Llama-3.1-Tulu-3-8B-SFT_ultra-feedback_08-06_11-52
# CUDA_VISIBLE_DEVICES=2,3 bash ./project_commands.sh

# cd ../gpu_burn
# make CUDAPATH=/usr/local/cuda-13.0
# ./gpu_burn 36000000000

# cd ./talas_vlm_embed
# bash ./project_commands.sh

# cd ./segd_distillation
# bash ./project_commands.sh

# cd ./gen_data
# cd deepseek_output
# mkdir -p split

# for f in *.zip; do
#     split -b 24M -d -a 3 "$f" "split/${f}.part_"
# done

# ls -R
# bash ./project_commands.sh

# cd ./reward-guidance-main
# bash ./project_command.sh
