# PAD Training

README này hướng dẫn chạy PAD với student model và teacher model tùy chọn.

Pipeline gồm ba bước:

1. Student sinh các responses.
2. Teacher chấm responses và tạo preference probabilities.
3. PAD train student trên dataset đã được teacher chấm.

## 1. Cài đặt

Chạy từ thư mục `PAD`:

```bash
cd /media/volume/ES_volumne/dat/PAD
mamba env create -f environment.yml
mamba activate pad
```

Nếu dùng model Hugging Face private hoặc gated:

```bash
huggingface-cli login
```

## 2. Chọn model và tạo dataset

`STUDENT_MODEL` và `TEACHER_MODEL` có thể là Hugging Face ID hoặc đường dẫn
local tuyệt đối.

```bash
export STUDENT_MODEL=/absolute/path/to/student-model
export TEACHER_MODEL=/absolute/path/to/teacher-model
export TEACHER_ID=my-teacher
export STUDENT_DIR=data/generated/ultrafeedback/my-student
```

Ví dụ dùng Hugging Face ID:

```bash
export STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Sinh responses bằng student:

```bash
bash data_gen/scripts/sampling.sh
```

Teacher chấm responses và tạo PAD dataset:

```bash
bash data_gen/scripts/pipeline_n4_gemma.sh
```

Dataset sau bước này nằm tại:

```text
$STUDENT_DIR/pkd-dataset-teacher-$TEACHER_ID-n4
```

## 3. Cấu hình training

Mở [`training_configs/gemma-2-2b-it-pd.yaml`](training_configs/gemma-2-2b-it-pd.yaml)
và đặt `model_name_or_path` là đúng student model đã dùng ở bước trên.

Ví dụ:

```yaml
model_name_or_path: /absolute/path/to/student-model

dataset_mixer:
  data/generated/ultrafeedback/my-student/pkd-dataset-teacher-my-teacher-n4: 1.0
dataset_splits:
- train
- test
local_dataset: true
```

Nếu dùng Hugging Face ID cho student thì đặt cùng ID trong YAML:

```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
```

Không cần đặt `TEACHER_MODEL` trong YAML. Teacher chỉ được dùng ở bước tạo
dataset; lúc training PAD đọc probabilities đã lưu trong dataset.

## 4. Train

Sau khi đã tạo dataset và cập nhật YAML:

```bash
bash run_ppd.sh
```

Checkpoint và model cuối được lưu theo `output_dir` trong YAML, mặc định:

```text
outputs/gemma-2-2b-it-pd
```

## Lưu ý

- `STUDENT_DIR` phải giống nhau khi chạy `sampling.sh` và
  `pipeline_n4_gemma.sh`.
- Nếu đổi teacher, phải chạy lại cả bước sinh responses và tạo dataset, sau đó
  cập nhật lại `dataset_mixer` trong YAML.
- Nếu đổi student, phải cập nhật cả `STUDENT_MODEL` và
  `model_name_or_path` trong YAML.
- `run_ppd.sh` không tự tạo dataset và không tự load teacher.
