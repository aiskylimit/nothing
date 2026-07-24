data_type=$1
model_name=$2
checkpoint_step=$3
dataset=$4
TEMPLATE="${TEMPLATE:-llama2}"

model_name_or_path="huggingface_models/${model_name}"

if [ "$dataset" == "example_text2sql_train" ]; then
    task_name="${model_name}"
elif [ "$dataset" == "example_text2sql_train_with_evidence" ]; then
    task_name="${model_name}_with_evidence"
fi

if [ "$checkpoint_step" == "checkpoint_lastest" ]; then
    checkpoint_dir="dbgpt_hub/output/adapter_kd/${data_type}/${task_name}"
else
    checkpoint_dir="dbgpt_hub/output/adapter_kd/${data_type}/${task_name}/${checkpoint_step}"
fi

output_path=dbgpt_hub/output/merged/${data_type}/${task_name}/
mkdir -p ${output_path}


python dbgpt_hub/train/export_model.py \
    --model_name_or_path $model_name_or_path \
    --template ${TEMPLATE} \
    --finetuning_type lora \
    --checkpoint_dir $checkpoint_dir \
    --output_dir $output_path \
    --fp16

echo 'merged model output path:' ${output_path}
