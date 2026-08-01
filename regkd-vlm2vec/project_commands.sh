#!/usr/bin/env bash
set -e

# Run this file after:
#   cd regkd-vlm2vec

# =========================
# 1. Optional system setup
# =========================
# Uncomment these lines if this is a fresh machine and you have sudo access.
#
# sudo apt-get update
# sudo apt-get upgrade -y


# =========================
# 2. Create Python env and install requirements
# =========================
export UV_PROJECT_ENVIRONMENT=vlm
uv sync
source vlm/bin/activate


# =========================
# 3. Optional eval images
# =========================
# README says this step is optional.
# Uncomment if you need eval images.

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip -o images.zip -d eval_images/
rm -rf images.zip

# =========================
# 4. Optional train images
# =========================
# This can take more than 1 hour.
# Uncomment if you need train images.
#
# bash download_traindata.sh
# bash download_traindata_2.sh

python download.py

# =========================
# 5. Optional teacher gradients
# =========================
# Download this file first:
#   https://huggingface.co/dangnguyens1/teacher_gradients/blob/main/qwenvl_2b_cls_vqa_grad.zip
#
# Then put it in this project folder and uncomment:
# unzip qwenvl_2b_cls_vqa_grad.zip
# mv /mnt/disk1/backup_user/dang.nh4/VLM_Embed/teacher_gradients ./teacher_gradients


wget https://huggingface.co/dangnguyens1/teacher_gradients/resolve/main/qwenvl_2b_cls_vqa_grad.zip
unzip qwenvl_2b_cls_vqa_grad.zip
mv ./mnt/disk1/backup_user/dang.nh4/VLM_Embed/teacher_gradients ./teacher_gradients
rm -rf qwenvl_2b_cls_vqa_grad.zip 


# =========================
# 6. Fix transformers code
# =========================
# README says this fixes the qwen2_vl image processor issue.
python fix_lib.py


# =========================
# 7. Train
# =========================
# Before running, check these args in scripts/test_gvendi.sh:
#   --image_dir
#   --teacher_cache_dir
#
# Uncomment to start training.
#
# bash scripts/test_gvendi.sh

# bash rebuttal_scripts/train_phase2_fastvlm_cls_directGrad.sh &
# bash rebuttal_scripts/train_phase2_fastvlm_cls_GradKD_only.sh &
bash rebuttal_scripts/train_phase2_fastvlm_cls_phrase1_K50.sh &
# bash rebuttal_scripts/train_phase2_fastvlm_cls_phrase1_K80.sh &
# bash rebuttal_scripts/train_phase2_fastvlm_cls_phrase1_K100.sh &
# wait



# =========================
# 8. Eval
# =========================
# Run 4 eval scripts in parallel for each batch size, each one on a different GPU.
# Override this list when needed, for example:
#   EVAL_BATCH_SIZES="24 12 6 3" bash project_commands.sh
EVAL_BATCH_SIZES="${EVAL_BATCH_SIZES:-32 28 24 16 10}"

for EVAL_BATCH_SIZE in ${EVAL_BATCH_SIZES}; do
    export EVAL_BATCH_SIZE
    echo "Running eval with batch size ${EVAL_BATCH_SIZE}"

    # CUDA_VISIBLE_DEVICES=4 bash eval_scripts/eval_phase2_fastvlm_cls_directGrad.sh &

    # CUDA_VISIBLE_DEVICES=5 bash eval_scripts/eval_phase2_fastvlm_cls_GradKD_only.sh &

    CUDA_VISIBLE_DEVICES=5 bash eval_scripts/eval_phase2_fastvlm_cls_phrase1_K50.sh &

    # CUDA_VISIBLE_DEVICES=6 bash eval_scripts/eval_phase2_fastvlm_cls_phrase1_K80.sh &

    # CUDA_VISIBLE_DEVICES=7 bash eval_scripts/eval_phase2_fastvlm_cls_phrase1_K100.sh &

    wait
done


# =========================
# 9. Copy JSON eval outputs
# =========================

JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json}"

python json_filter.py ./MMEB-evaloutputs "${JSON_FILTER_DESTINATION}" --overwrite
