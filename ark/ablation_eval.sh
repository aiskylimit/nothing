export HF_HOME="/mnt/data/huggingface/"
export HUGGINGFACE_HUB_DISABLE_UPDATE_CHECK=1

cuda_device=${1-6}

# num sink tokens
for sink_tokens in 0 1 8
do
CUDA_VISIBLE_DEVICES=$cuda_device python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,cache_sink_tokens=${sink_tokens} \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 24 \
  --output_path results/ablation/sink-tokens-${sink_tokens}.json
done

# num sliding window tokens
for sliding_window in 0 16 64
do
CUDA_VISIBLE_DEVICES=$cuda_device python evaluation.py --model octopus \
  --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,cache_recent_window=${sliding_window} \
  --tasks gsm8k --num_fewshot 5 \
  --device cuda --batch_size 24 \
  --output_path results/ablation/sliding-window-${sliding_window}.json
done
