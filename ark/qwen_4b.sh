# export HF_HOME="/mnt/data/huggingface/"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

num_gpu=4
batch_size=2
grad_accum=2
cuda_device="4,5,6,7"

###############################################################
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir checkpoints/qwen-4b-alpaca-cleaned-octopus

# eval
CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned-octopus \
  --tasks hellaswag --num_fewshot 0 \
  --device cuda --batch_size 32 \
  --output_path results/octopus/qwen-4b-alpaca-cleaned-octopus-hellaswag.json

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned-octopus \
  --tasks arc_easy --num_fewshot 0 \
  --device cuda --batch_size 16 \
  --output_path results/octopus/qwen-4b-alpaca-cleaned-octopus-arc-easy.json

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned-octopus \
  --tasks arc_challenge --num_fewshot 0 \
  --device cuda --batch_size 16 \
  --output_path results/octopus/qwen-4b-alpaca-cleaned-octopus-arc-challenge.json

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned-octopus \
  --tasks mmlu --num_fewshot 5 \
  --device cuda --batch_size 4 \
  --output_path results/octopus/qwen-4b-alpaca-cleaned-octopus-mmlu.json

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-4b-alpaca-cleaned-octopus \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 24 \
  --output_path results/octopus/qwen-4b-alpaca-cleaned-octopus-gsm8k.json