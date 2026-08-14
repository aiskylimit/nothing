#!/bin/zsh

model_name_1="/media/volume/ES_volumne/dat/Gen-DPO/output/dpo_Qwen2.5-1.5B-SFT-UltraChat-merged_ultra-feedback_07-23_13-17"
model_name_2="/media/volume/ES_volumne/dat/Gen-DPO/output/dpo_Qwen2.5-1.5B-SFT-UltraChat-merged_ultra-feedback_reverse_07-23_23-49"
input_dir="/media/volume/ES_volumne/dat/Gen-DPO/datasets/ultra-feedback"
output_dir="/media/volume/ES_volumne/dat/Gen-DPO/datasets/ultra-feedback-tisdpo"
model1_template="normal"
model2_template="normal"
batch_size=8
num_gpus=1
force_sequential=false  # Set to true if multiprocessing causes issues

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Run the parallel processing script
python /media/volume/ES_volumne/dat/Gen-DPO/token_weight_estimation.py \
  --model_name_1 $model_name_1 \
  --model_name_2 $model_name_2 \
  --model1_template $model1_template \
  --model2_template $model2_template \
  --input_dir $input_dir \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $(if $force_sequential; then echo "--force_sequential"; fi) 