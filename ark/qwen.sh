export HF_HOME="/mnt/data/huggingface/"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

num_gpu=4
batch_size=1
grad_accum=4
cuda_device="4,5,6,7"

###############################################################
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --model-name Qwen/Qwen3-1.7B \
  --dataset-name open-r1/Mixture-of-Thoughts --max-seq-length 8192 --max-sample 5000 \
  --output-dir checkpoints/qwen-1.7b-mot

# eval

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-1.7b-mot,max_length=35000,cache_budget=1024 \
  --tasks aime24 --num_fewshot 0 \
  --device cuda --batch_size 4 \
  --apply_chat_template \
  --output_path results/qwen-1.7b-mot-aime24-1k.json &

CUDA_VISIBLE_DEVICES=5 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-1.7b-mot,max_length=35000,cache_budget=2048 \
  --tasks aime24 --num_fewshot 0 \
  --device cuda --batch_size 3 \
  --apply_chat_template \
  --output_path results/qwen-1.7b-mot-aime24-2k.json &

CUDA_VISIBLE_DEVICES=6 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/qwen-1.7b-mot,max_length=35000,cache_budget=4096 \
  --tasks aime24 --num_fewshot 0 \
  --device cuda --batch_size 3 \
  --apply_chat_template \
  --output_path results/qwen-1.7b-mot-aime24-4k.json &

wait

# lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen3-1.7B,max_length=35000 \
#     --tasks hellaswag --num_fewshot 0 \
#     --device cuda:0 --batch_size 4 \
#     --apply_chat_template \
#     --output_path results/base_model/qwen-aime24.json
