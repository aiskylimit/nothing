uv sync
source .venv/bin/activate

# hf download Dream-AI-HUST/sql_benchmarks \
#   --repo-type dataset \
#   --local-dir .
# unzip benchmarks.zip
# unzip data.zip
python ./scripts/synid_augment/build_teacher_train_from_final_merged.py

RUNNER_GPU_LIST=0,1,2,3,4,5,6,7 \
GPUS_PER_JOB=4 \
RUN_MODE=parallel \
SKIP_EXISTING=false \
INFER_SEEDS=10,42,50,100,1234 \
EVAL_BATCH_SIZE=32 \
INFER_BATCH_SIZE=128 \
bash scripts/qwen/synid_ce_multilayer_3/run_full_pipeline.sh

RUNNER_GPU_LIST=0,1,2,3,4,5,6,7 \
GPUS_PER_JOB=4 \
RUN_MODE=parallel \
SKIP_EXISTING=false \
INFER_SEEDS=10,42,50,100,1234 \
EVAL_BATCH_SIZE=32 \
INFER_BATCH_SIZE=128 \
bash scripts/qwen/synid_ce_no_keywords_weight/run_full_pipeline.sh