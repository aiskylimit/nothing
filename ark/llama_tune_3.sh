export HF_HOME="/mnt/data/huggingface/"
export MASTER_PORT_1=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
# change num of gpus and gpu ids
# modify grad accum, then batch size; ensure global batch size = batch size * grad accum * num gpus = 16
num_gpu=8
batch_size=2
grad_accum=1
cuda_device="0,1,2,3,4,5,6,7"

# longer training, score per kv head and forward kl, higher kd ratio
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT_1 --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --num-epochs 5 --phase1-epochs 2 --kl-type fkl \
  --sink-token-value-threshold 20 --kd-ratio 0.5 \
  --output-dir checkpoints/llama-8b-alpaca-cleaned-1

# eval
CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 32 \
  --output_path results/score_kv_head/fkl/llama-8b-alpaca-cleaned-gsm8k.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,max_length=3200,sink_token_value_threshold=20 \
  --tasks mmlu --num_fewshot 5 \
  --device cuda --batch_size 2 \
  --output_path results/score_kv_head/fkl/llama-8b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks hellaswag --num_fewshot 0 \
  --device cuda --batch_size 32 \
  --output_path results/score_kv_head/fkl/llama-8b-alpaca-cleaned-hellaswag.json

# longer training, score per kv head and reverse kl, higher kd ratio
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT_1 --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --num-epochs 5 --phase1-epochs 2 --kl-type rkl \
  --sink-token-value-threshold 20 --kd-ratio 0.5 \
  --output-dir checkpoints/llama-8b-alpaca-cleaned-1

# eval
CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 32 \
  --output_path results/score_kv_head/rkl/llama-8b-alpaca-cleaned-gsm8k.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,max_length=3200,sink_token_value_threshold=20 \
  --tasks mmlu --num_fewshot 5 \
  --device cuda --batch_size 2 \
  --output_path results/score_kv_head/rkl/llama-8b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks hellaswag --num_fewshot 0 \
  --device cuda --batch_size 32 \
  --output_path results/score_kv_head/rkl/llama-8b-alpaca-cleaned-hellaswag.json

# full distillation
# longer training, score per kv head and forward kl, higher kd ratio
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT_1 --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --num-epochs 5 --phase1-epochs 5 --kl-type fkl \
  --sink-token-value-threshold 20 --kd-ratio 0.5 \
  --output-dir checkpoints/llama-8b-alpaca-cleaned-1

# eval
CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 32 \
  --output_path results/score_kv_head/fkl/llama-8b-alpaca-cleaned-gsm8k.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,max_length=3200,sink_token_value_threshold=20 \
  --tasks mmlu --num_fewshot 5 \
  --device cuda --batch_size 2 \
  --output_path results/score_kv_head/fkl/llama-8b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks hellaswag --num_fewshot 0 \
  --device cuda --batch_size 32 \
  --output_path results/score_kv_head/fkl/llama-8b-alpaca-cleaned-hellaswag.json

# longer training, score per kv head and reverse kl, higher kd ratio
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT_1 --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --num-epochs 5 --phase1-epochs 5 --kl-type rkl \
  --sink-token-value-threshold 20 --kd-ratio 0.5 \
  --output-dir checkpoints/llama-8b-alpaca-cleaned-1

# eval
CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 32 \
  --output_path results/score_kv_head/rkl/llama-8b-alpaca-cleaned-gsm8k.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,max_length=3200,sink_token_value_threshold=20 \
  --tasks mmlu --num_fewshot 5 \
  --device cuda --batch_size 2 \
  --output_path results/score_kv_head/rkl/llama-8b-alpaca-cleaned-mmlu.json

CUDA_VISIBLE_DEVICES=${cuda_device} python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned-1,dtype=bfloat16,sink_token_value_threshold=20 \
  --tasks hellaswag --num_fewshot 0 \
  --device cuda --batch_size 32 \
  --output_path results/score_kv_head/rkl/llama-8b-alpaca-cleaned-hellaswag.json
