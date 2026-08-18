# PAD: thay student model và teacher model

README này mô tả cách chạy PAD khi muốn tự chọn cả student model và teacher
model. Hai model được dùng ở hai bước khác nhau:

1. Student sinh nhiều câu trả lời ứng viên.
2. Teacher chấm các ứng viên và tạo ra các xác suất preference.
3. PAD train student bằng dataset đã chứa xác suất do teacher tạo ra.

Vì vậy, `run_ppd.sh` không load teacher trực tiếp. Muốn đổi teacher, bắt buộc
phải tạo lại local PAD dataset trước khi train.

## 1. Cài đặt môi trường

Tạo environment:

```sh
mamba env create -f environment.yml
```

Activate environment:

```sh
mamba activate pad
```

## 2. Chuẩn bị

Chạy các lệnh từ thư mục `PAD`:

```sh
cd /media/volume/ES_volumne/dat/PAD
```

Nếu model hoặc dataset trên Hugging Face là private/gated:

```sh
huggingface-cli login
```

## 3. Dataset mặc định

Mặc định codebase train trực tiếp từ dataset Hugging Face:

```text
pvdhihihi/ultra-feedback
```

The active training config is:

```text
training_configs/gemma-2-2b-it-pd.yaml
```

The relevant dataset settings are:

```yaml
dataset_mixer:
  pvdhihihi/ultra-feedback: 1.0
dataset_splits:
- train_prefs
- test_prefs
local_dataset: false
```

Dataset này đã chứa teacher probabilities. Do đó, nếu dùng dataset mặc định,
việc set `TEACHER_MODEL` sẽ không thay đổi teacher đã dùng để tạo dataset.

Dataset và model fallback được khai báo trong:

```text
training_configs/gemma-2-2b-it-pd.yaml
```

Nếu dataset dùng split `train` và `test` thay vì `train_prefs` và `test_prefs`,
đổi trong YAML:

```yaml
dataset_splits:
- train
- test
```

## 4. Workflow đổi cả student và teacher

### Bước 1: Chọn student và sinh responses

Đặt path hoặc Hugging Face ID của student qua `STUDENT_MODEL`. Đặt
`STUDENT_DIR` là thư mục trung gian dùng chung cho toàn bộ pipeline:

```sh
export STUDENT_MODEL=/absolute/path/to/student-model
# Ví dụ HF ID:
# export STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct

export STUDENT_DIR=data/generated/ultrafeedback/my-student

bash data_gen/scripts/sampling.sh
```

`sampling.sh` sẽ sinh responses vào `$STUDENT_DIR` với nhiều seed.

### Bước 2: Chọn teacher và tạo PAD dataset

Đặt path hoặc Hugging Face ID của teacher:

```sh
export TEACHER_MODEL=/absolute/path/to/teacher-model
# Ví dụ HF ID:
# export TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct

export TEACHER_ID=my-teacher

bash data_gen/scripts/pipeline_n4_gemma.sh
```

Pipeline sẽ tạo dataset tại:

```text
$STUDENT_DIR/pkd-dataset-teacher-$TEACHER_ID-n4
```

Teacher được dùng trong `prob.py` và `prob_sl.py` để tính preference scores.

### Bước 3: Trỏ training config tới dataset vừa tạo

Mở `training_configs/gemma-2-2b-it-pd.yaml` và đổi:

```yaml
dataset_mixer:
  data/generated/ultrafeedback/my-student/pkd-dataset-teacher-my-teacher-n4: 1.0
local_dataset: true
dataset_splits:
- train
- test
```

Giữ `model_name_or_path` làm fallback student model, hoặc override student
trực tiếp khi chạy train:

```sh
export STUDENT_MODEL=/absolute/path/to/student-model
bash run_ppd.sh
```

Khi đã chạy bước 1 và 2, lệnh train đầy đủ là:

```sh
export STUDENT_MODEL=/absolute/path/to/student-model
export TEACHER_MODEL=/absolute/path/to/teacher-model
export TEACHER_ID=my-teacher
export STUDENT_DIR=data/generated/ultrafeedback/my-student

bash data_gen/scripts/sampling.sh
bash data_gen/scripts/pipeline_n4_gemma.sh

# Sau đó sửa dataset_mixer/local_dataset trong YAML như ở trên.
bash run_ppd.sh
```

The trained model will be saved under:

```text
outputs/*
```

### Các biến và nơi chúng được dùng

| Biến | Được dùng ở đâu | Ý nghĩa |
|---|---|---|
| `STUDENT_MODEL` | `sampling.sh`, `run_ppd.sh` | Student sinh responses và được train |
| `TEACHER_MODEL` | `pipeline_n4_gemma.sh` | Teacher chấm responses |
| `TEACHER_ID` | `pipeline_n4_gemma.sh` | Tên để tạo output directory |
| `STUDENT_DIR` | `sampling.sh`, `pipeline_n4_gemma.sh` | Thư mục dữ liệu trung gian |

Các field quan trọng trong `training_configs/gemma-2-2b-it-pd.yaml`:

- `model_name_or_path`: fallback local path or Hugging Face id for the student model.
- `dataset_mixer`: dataset id to train on.
- `output_dir`: where checkpoints and final outputs are written.

Student có thể override mà không sửa YAML:

```sh
STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct bash run_ppd.sh
```

Local paths are also supported:

```sh
STUDENT_MODEL=/absolute/path/to/student-model bash run_ppd.sh
```

Local path và Hugging Face ID đều được hỗ trợ cho cả hai model.

## 5. Output và lưu ý

Model sau khi train được lưu theo `output_dir` trong YAML, mặc định dưới:

```text
outputs/*
```

Không cần tạo lại dataset nếu chỉ đổi student nhưng vẫn muốn dùng đúng teacher
probabilities của dataset hiện tại. Tuy nhiên, nếu đổi teacher thì phải chạy
lại toàn bộ bước 1--3.

Để quay lại dataset Hugging Face mặc định, đặt lại:

```yaml
dataset_mixer:
  pvdhihihi/ultra-feedback: 1.0
local_dataset: false
```

## Evaluation

We follow the official implementation for evaluation on AlpacaEval 2, Arena-Hard, MT-Bench and GSM8K.

* AlpacaEval 2: Please refer to the [AlpacaEval repo](https://github.com/tatsu-lab/alpaca_eval) for evaluation.

* Arena-Hard: Please refer to to the [Arena-Hard-Auto repo](https://github.com/lm-sys/arena-hard-auto) for evaluation.

* MT-Bench: Please refer to the [FastChat repo](https://github.com/lm-sys/FastChat) for evaluation.

* GSM8K: Please refer to the [ZeroEval repo](https://github.com/WildEval/ZeroEval) for evaluation.


## Training Report

The report below is from the original Gemma PAD experiment and is kept as a reference.

### Overview
This part contains training logs and comparative analysis of three preference alignment methods: SimPO, DPO, and PAD. We document the training process, implementation details, and performance metrics for each approach.

### Implementations
- **DPO**: Based on the implementation from [TRL](https://github.com/huggingface/trl)
- **SimPO**: Based on the implementation from [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO)

### Training Configuration

#### Models
- **Student Model**: Gemma-2-2B-It
- **Teacher Model**: Gemma-2-9B-It

#### Hardware
- **GPUs**: 2 × A800 (80G)

#### Training Parameters
- **Training Type**: Full parameter fine-tuning
- **Memory Optimization**: ZeRO Stage 2
- **Epochs**: 1
- **Precision**: BFloat16
- **Dataset Size**:
  - Training samples: 55,321
  - Test samples: 1,130
- **Batch Size**: 128
- **Total Training Steps**: 432
- **Maximum Sequence Length**: 2048
- **Per Device Train Batch Size**: 2
- **Per Device Evaluation Batch Size**: 2
- **Gradient Accumulation Steps**: 32
- **Evaluation Frequency**: Every 100 training steps
- **Gradient Checkpointing**: Enabled

For additional parameters, please refer to the paper or the configuration files.

### Results

| Method | GPU Hours | Alpaca-Eval 2.0 LC (%) |
|--------|-----------|---------------------------------|
| DPO    | 8.7856    | 43.77                           |
| SimPO  | 7.2672    | 44.94                           |
| PAD    | 7.2884    | 45.73                           |

You can find the training log under `gemma-log/*`.

#### Analysis
- **Training Efficiency**: PAD and SimPO require similar computational resources, while DPO demands notably more. This efficiency difference is primarily because DPO requires loading an additional reference model during training, whereas PAD and SimPO do not.
- **Performance**: PAD outperforms both SimPO and DPO in terms of win rate, which aligns with the findings reported in the submission paper.
