export HF_HOME="/mnt/data/huggingface/"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
# change num of gpus and gpu ids
# modify grad accum, then batch size; ensure global batch size = batch size * grad accum * num gpus = 16
num_gpu=1
batch_size=4
grad_accum=4
cuda_device="0"




# no distillation (kd_ratio = 0)
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=$num_gpu --master_port $MASTER_PORT --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --no-distill --sink-token-value-threshold 20

# eval
CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 32 \
  --output_path results/no-distill/llama-8b-alpaca-cleaned-gsm8k.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,dtype=bfloat16,max_length=3200,sink_token_value_threshold=20 \
  --tasks mmlu --num_fewshot 5 \
  --device cuda --batch_size 2 \
  --output_path results/no-distill/llama-8b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks hellaswag --num_fewshot 0 \
  --device cuda --batch_size 32 \
  --output_path results/no-distill/llama-8b-alpaca-cleaned-hellaswag.json

# kd ratio = 0.5
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=$num_gpu --master_port $MASTER_PORT --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --kd-ratio 0.5 --sink-token-value-threshold 20

# eval
CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 32 \
  --output_path results/kd-ratio-0.5/llama-8b-alpaca-cleaned-gsm8k.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,dtype=bfloat16,max_length=3200,sink_token_value_threshold=20 \
  --tasks mmlu --num_fewshot 5 \
  --device cuda --batch_size 2 \
  --output_path results/kd-ratio-0.5/llama-8b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks hellaswag --num_fewshot 0 \
  --device cuda --batch_size 32 \
  --output_path results/kd-ratio-0.5/llama-8b-alpaca-cleaned-hellaswag.json
