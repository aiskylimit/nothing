source .venv/bin/activate

hf download Dream-AI-HUST/test --repo-type dataset --local-dir data

bash ./start_rl_training.sh --config_file config/deepspeed/zero2_4GPU.yaml --config-name qwen3.5_9b.yaml