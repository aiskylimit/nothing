export HF_HOME="/mnt/data/huggingface/"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

num_gpu=4
batch_size=2
grad_accum=2
cuda_device="4,5,6,7"

for reg_weight in 0.0 0.25 0.75 1.0 1.5 2.0
do
CUDA_VISIBLE_DEVICES=${cuda_device} torchrun \
  --nproc_per_node=${num_gpu} --master_port $MASTER_PORT --local-ranks-filter 0 \
  -m octopus.train \
  --batch-size $batch_size --grad-accum $grad_accum \
  --regularization-weight ${reg_weight}

CUDA_VISIBLE_DEVICES=4 python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 24 \
  --output_path results/ablation/reg-weight-${reg_weight}.json

done
