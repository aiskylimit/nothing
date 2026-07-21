export HF_HOME="/mnt/data/huggingface/"
export MASTER_PORT_1=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
# change num of gpus and gpu ids
# modify grad accum, then batch size; ensure global batch size = batch size * grad accum * num gpus = 16
num_gpu=1
batch_size=4
grad_accum=4
cuda_device="0"

# longer training: 4 epochs, 2 each phase
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT_1 --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --num-epochs 4 --phase1-epochs 2 \
  --sink-token-value-threshold 20 \
  --output-dir checkpoints/llama-8b-alpaca-cleaned-1

# eval
CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 32 \
  --output_path results/longer/llama-8b-alpaca-cleaned-gsm8k.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,max_length=3200,sink_token_value_threshold=20 \
  --tasks mmlu --num_fewshot 5 \
  --device cuda --batch_size 2 \
  --output_path results/longer/llama-8b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks hellaswag --num_fewshot 0 \
  --device cuda --batch_size 32 \
  --output_path results/longer/llama-8b-alpaca-cleaned-hellaswag.json

# separate to another file for parallel exec, keep all env vars above
# longer training and forward kl
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT_1 --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --num-epochs 4 --phase1-epochs 2 --kl-type fkl \
  --sink-token-value-threshold 20 \
  --output-dir checkpoints/llama-8b-alpaca-cleaned-1

# eval
CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 32 \
  --output_path results/longer_fkl/llama-8b-alpaca-cleaned-gsm8k.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,max_length=3200,sink_token_value_threshold=20 \
  --tasks mmlu --num_fewshot 5 \
  --device cuda --batch_size 2 \
  --output_path results/longer_fkl/llama-8b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks hellaswag --num_fewshot 0 \
  --device cuda --batch_size 32 \
  --output_path results/longer_fkl/llama-8b-alpaca-cleaned-hellaswag.json
