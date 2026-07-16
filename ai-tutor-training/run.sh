# ./start_rl_training.sh --config_file config/deepspeed/1GPU.yaml --config-name 7b.yaml

# accelerate   launch   --config_file   config/deepspeed/1GPU.yaml   train_rl.py   --config-name   qwen3.5_9b.yaml

bash ./start_rl_training.sh --config_file config/deepspeed/zero2_8GPU.yaml --config-name qwen3.5_9b.yaml