export HF_HOME="/mnt/data/huggingface/"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
torchrun --nproc_per_node=1 --master_port $MASTER_PORT --local-ranks-filter 0 -m octopus.train
