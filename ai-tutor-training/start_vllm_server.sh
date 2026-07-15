#!/usr/bin/env bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

./stop_vllm_server.sh

python vllm_server.py "$@"



# #-----------------------------------------
# # Start the VLLM server
# #-----------------------------------------
# echo "[start_rl_training.sh] Launching VLLM server..."
# ./stop_vllm_server.sh || true
# sleep 2
# python vllm_server.py --config-name   7b.yaml &
# SERVER_PID=$!

# #-----------------------------------------
# # Wait until the server responds
# #-----------------------------------------
# until curl -fsS http://localhost:8005/docs >/dev/null ; do
#   echo "[start_rl_training.sh] Waiting for VLLM server..."
#   sleep 5
# done
# echo "[start_rl_training.sh] VLLM server is up."
