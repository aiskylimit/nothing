data_type=$1
task_name=$2
model_name_or_path=$3
checkpoint_step=$4
dataset=$5

checkpoint_output=$checkpoint_step

output_path=dbgpt_hub/output/adapter_kd/${data_type}/${task_name}/preds/${checkpoint_output}/
mkdir -p ${output_path}


for i in `seq 0 7`
do
    bash scripts/predict_sft_split.sh ${i} ${data_type} ${i}:8 ${task_name} ${model_name_or_path} ${checkpoint_output} ${dataset} &
done
wait

cat ${output_path}/pred_codes-*.sql > ${output_path}/pred_codes.sql 
rm -r ${output_path}/pred_codes-*.sql


if [ "$data_type" == "bird" ]; then
    python dbgpt_hub/eval/evaluation_bird.py --predicted_sql_path ${output_path}/pred_codes.sql --ground_truth_path dbgpt_hub/data/bird/dev/dev.sql  --db_root_path dbgpt_hub/data/bird/dev/dev_databases/ --diff_json_path dbgpt_hub/data/bird/dev/dev.json  >> ${output_path}/pred_codes.result
else 
    python dbgpt_hub/eval/evaluation.py --plug_value --input ${output_path}/pred_codes.sql >> ${output_path}/pred_codes.result
fi
