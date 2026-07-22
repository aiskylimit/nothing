# export HF_HOME="/mnt/data/huggingface/"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

num_gpu=4
batch_size=2
grad_accum=2
cuda_device="4,5,6,7"

###############################################################
# Phase 1 only
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir checkpoints/qwen-4b-alpaca-cleaned-phase1

# eval
# CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
#   --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned \
#   --tasks hellaswag --num_fewshot 0 \
#   --device cuda --batch_size 32 \
#   --output_path results/phase-1/qwen-4b-alpaca-cleaned-hellaswag.json

# CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
#   --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned \
#   --tasks arc_easy --num_fewshot 0 \
#   --device cuda --batch_size 16 \
#   --output_path results/phase-1/qwen-4b-alpaca-cleaned-arc-easy.json

# CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
#   --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned \
#   --tasks arc_challenge --num_fewshot 0 \
#   --device cuda --batch_size 16 \
#   --output_path results/phase-1/qwen-4b-alpaca-cleaned-arc-challenge.json

# CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
#   --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned \
#   --tasks mmlu --num_fewshot 5 \
#   --device cuda --batch_size 4 \
#   --output_path results/phase-1/qwen-4b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned-phase1 \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 24 \
  --output_path results/phase-1/qwen-4b-alpaca-cleaned-gsm8k.json


# Phase 1 + 2
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --num-epochs 3 \
  --output-dir checkpoints/qwen-4b-alpaca-cleaned-2phase

# eval
# CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
#   --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned \
#   --tasks hellaswag --num_fewshot 0 \
#   --device cuda --batch_size 32 \
#   --output_path results/2-phase/qwen-4b-alpaca-cleaned-hellaswag.json

# CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
#   --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned \
#   --tasks arc_easy --num_fewshot 0 \
#   --device cuda --batch_size 16 \
#   --output_path results/2-phase/qwen-4b-alpaca-cleaned-arc-easy.json

# CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
#   --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned \
#   --tasks arc_challenge --num_fewshot 0 \
#   --device cuda --batch_size 16 \
#   --output_path results/2-phase/qwen-4b-alpaca-cleaned-arc-challenge.json

# CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
#   --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned \
#   --tasks mmlu --num_fewshot 5 \
#   --device cuda --batch_size 4 \
#   --output_path results/2-phase/qwen-4b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned-2phase \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 24 \
  --output_path results/2-phase/qwen-4b-alpaca-cleaned-gsm8k.json


# eval Qwen3 4B Inst base
# lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=2048 \
#     --tasks hellaswag --num_fewshot 0 \
#     --device cuda --batch_size 8 \
#     --output_path results/base_model/qwen-4b-hellaswag.json

# lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=2048 \
#     --tasks arc_easy --num_fewshot 0 \
#     --device cuda --batch_size 8 \
#     --output_path results/base_model/qwen-4b-arc-easy.json

# lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=2048 \
#     --tasks arc_challenge --num_fewshot 0 \
#     --device cuda --batch_size 8 \
#     --output_path results/base_model/qwen-4b-arc-challenge.json

# lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=2048 \
#     --tasks mmlu --num_fewshot 5 \
#     --device cuda --batch_size 4 \
#     --output_path results/base_model/qwen-4b-mmlu.json

# lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,max_length=2048 \
#     --tasks gsm8k --num_fewshot 5 \
#     --device cuda --batch_size 32 \
#     --output_path results/base_model/qwen-4b-gsm8k.json
