# export HF_HOME="/mnt/data/huggingface/"
export HUGGINGFACE_HUB_DISABLE_UPDATE_CHECK=1

hf download bachthetrollface/qwen-1.7b-mot \
    --local-dir checkpoints/qwen-1.7b-mot

cuda_device=${1-0}

# CUDA_VISIBLE_DEVICES=$cuda_device lm_eval --model hf \
#     --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks longbench --device cuda --batch_size 32 \
#     --output_path results/longbench/base_model.json

# CUDA_VISIBLE_DEVICES=$cuda_device lm_eval --model hf \
#     --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,max_length=2048 \
#     --tasks longbench --device cuda --batch_size 32 \
#     --output_path results/longbench/base_model_2048.json

# CUDA_VISIBLE_DEVICES=$cuda_device lm_eval --model hf \
#     --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,max_length=2560 \
#     --tasks longbench --device cuda --batch_size 16 \
#     --output_path results/longbench/base_model_2560.json

CUDA_VISIBLE_DEVICES=$cuda_device lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-1.7B,max_length=8192 \
    --tasks longbench --device cuda --batch_size 16 \
    --output_path results/longbench/base_qwen_8192.json


# CUDA_VISIBLE_DEVICES=$cuda_device python evaluation.py --model octopus \
#   --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,cache_recent_window=16,max_length=2048,cache_budget=512 \
#   --tasks longbench --device cuda --batch_size 20 \
#   --output_path results/longbench/llama_2048.json

# CUDA_VISIBLE_DEVICES=$cuda_device python evaluation.py --model octopus \
#   --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,cache_recent_window=16,max_length=2560,cache_budget=640 \
#   --tasks longbench --device cuda --batch_size 12 \
#   --output_path results/longbench/llama_2560.json

CUDA_VISIBLE_DEVICES=$cuda_device python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-1.7b-mot,cache_recent_window=16,max_length=8192,cache_budget=2048 \
  --tasks longbench --device cuda --batch_size 4 \
  --output_path results/longbench/qwen_8192.json
