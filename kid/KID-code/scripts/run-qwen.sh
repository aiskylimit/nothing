student_model=Qwen1.5-0.5B
teacher_model=Qwen1.5-1.8B      ### Qwen1.5-1.8B / Qwen1.5-4B / Qwen1.5-7B
teacher_model_path=dbgpt_hub/output/merged/spider/Qwen1.5-1.8B
task_name=spider    ### spider / bird
dataset=example_text2sql_train  ### example_text2sql_train / example_text2sql_train_with_evidence

#student SFT
bash scripts/train_sft-qwen-teacher.sh $task_name $student_model 12138 $dataset 

#teacher SFT
bash scripts/train_sft-qwen-teacher.sh $task_name $teacher_model 12138 $dataset  

#forward KD
bash scripts/train_sft-qwen.sh $task_name $student_model 12138 $dataset $teacher_model_path $teacher_model forward gt

#reverse KD
bash scripts/train_sft-qwen.sh $task_name $student_model 12138 $dataset $teacher_model_path $teacher_model reverse gt

#ImitKD
bash scripts/train_sft-qwen.sh $task_name $student_model 12138 $dataset $teacher_model_path $teacher_model forward mix_request_gt

#f-distill KD
bash scripts/train_sft-qwen.sh $task_name $student_model 12138 $dataset $teacher_model_path $teacher_model tvd mix_request_teacher

#GKD-FKL
bash scripts/train_sft-qwen.sh $task_name $student_model 12138 $dataset $teacher_model_path $teacher_model forward student

#GKD-RKL
bash scripts/train_sft-qwen.sh $task_name $student_model 12138 $dataset $teacher_model_path $teacher_model reverse student

#GKD-JSD
bash scripts/train_sft-qwen.sh $task_name $student_model 12138 $dataset $teacher_model_path $teacher_model jsd student

#KID (Ours)
bash scripts/train_sft-qwen.sh $task_name $student_model 12138 $dataset $teacher_model_path reverse mask_student random
