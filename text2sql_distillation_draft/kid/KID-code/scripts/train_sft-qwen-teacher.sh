wandb offline # Close wandb
data_type=$1
model_name=$2
port=$3
dataset=$4

model_name_or_path="huggingface_models/${model_name}"

if [ "$dataset" == "example_text2sql_train" ]; then
    task_name="${model_name}"
elif [ "$dataset" == "example_text2sql_train_with_evidence" ]; then
    task_name="${model_name}_with_evidence"
fi
output_dir="dbgpt_hub/output/adapter_kd/${data_type}/${task_name}"
mkdir -p ${output_dir}

current_date=$(date +"%Y%m%d_%H%M")
train_log="${output_dir}/logs/train_sft_test_${current_date}.log"
mkdir -p ${output_dir}/logs
start_time=$(date +%s)
echo " Train Start time: $(date -d @$start_time +'%Y-%m-%d %H:%M:%S')" >>${train_log}
    
echo "############train end###############" >>${train_log}
echo "Train End time: $(date)" >>${train_log}
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
min=$(( (duration % 3600) / 60))
echo "Time elapsed: ${hour}  hour $min min " >>${train_log}

deepspeed --num_gpus 8 --master_port ${port}  dbgpt_hub/train/sft_train.py \
    --deepspeed dbgpt_hub/configs/ds_config.json \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --dataset $dataset \
    --dataset_dir "dbgpt_hub/data/${data_type}_codes" \
    --max_source_length 2048 \
    --max_target_length 256 \
    --template llama2 \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_target q_proj,v_proj \
    --output_dir $output_dir \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine_with_restarts \
    --logging_steps 50 \
    --save_steps 1000 \
    --learning_rate 2e-4 \
    --num_train_epochs 8 \
    --plot_loss \
    --bf16 2>&1 | tee ${train_log}


for file in $output_dir/checkpoint-*
do 
    rm -r ${file}/global_step*
    rm -r ${file}/*.pth
    rm -r ${file}/scheduler.pt
done

for checkpoint_step in checkpoint-1000 checkpoint-2000 checkpoint-3000 checkpoint-4000 checkpoint_lastest
do 
    bash scripts/predict_sft_merge.sh ${data_type} ${task_name} ${model_name_or_path} ${checkpoint_step} $dataset
done

bash scripts/export_merge.sh ${data_type} ${model_name} checkpoint_lastest ${dataset}