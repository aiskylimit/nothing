cuda=$1
data_type=$2
split_part=$3
task_name=$4
model_name_or_path=$5
checkpoint_step=$6
dataset=$7

checkpoint_dir=dbgpt_hub/output/adapter_kd/${data_type}/${task_name}/${checkpoint_step}
if [ "$checkpoint_step" == "checkpoint_lastest" ]; then
    checkpoint_dir=dbgpt_hub/output/adapter_kd/${data_type}/${task_name}
fi
output_path=dbgpt_hub/output/adapter_kd/${data_type}/${task_name}/preds/${checkpoint_step}

if [ "$dataset" == "example_text2sql_train_with_evidence" ]; then
    predicted_input_filename=dbgpt_hub/data/${data_type}_codes/example_text2sql_dev_with_evidence.json
else 
    predicted_input_filename=dbgpt_hub/data/${data_type}_codes/example_text2sql_dev.json
fi

echo 'predcited input filename:' ${predicted_input_filename}

CUDA_VISIBLE_DEVICES=${cuda}  python dbgpt_hub/predict/predict.py \
    --model_name_or_path $model_name_or_path \
    --template llama2 \
    --split_part ${split_part} \
    --finetuning_type lora \
    --predicted_input_filename $predicted_input_filename \
    --checkpoint_dir ${checkpoint_dir} \
    --predicted_out_filename ${output_path}/pred_codes-${split_part}.sql
