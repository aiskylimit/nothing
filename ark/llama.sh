export HF_HOME="/mnt/data/huggingface/"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

num_gpu=4
batch_size=2
grad_accum=8
cuda_device="4,5,6,7"

###############################################################
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum

# eval

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned \
  --tasks hellaswag --num_fewshot 0 \
  --device cuda --batch_size 32 \
  --output_path results/llama-8b-alpaca-cleaned-hellaswag.json

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned \
  --tasks arc_easy --num_fewshot 0 \
  --device cuda --batch_size 16 \
  --output_path results/llama-8b-alpaca-cleaned-arc-easy.json

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned \
  --tasks arc_challenge --num_fewshot 0 \
  --device cuda --batch_size 16 \
  --output_path results/llama-8b-alpaca-cleaned-arc-challenge.json

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned \
  --tasks mmlu --num_fewshot 5 \
  --device cuda --batch_size 4 \
  --output_path results/llama-8b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 32 \
  --output_path results/llama-8b-alpaca-cleaned-gsm8k.json

# lm_eval --model hf \
#     --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,max_length=2048 \
#     --tasks hellaswag --num_fewshot 0 \
#     --device cuda:0 --batch_size 8 \
#     --output_path results/base_model/hellaswag.json
