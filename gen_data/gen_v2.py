import os
import json
import zipfile
import math
import io
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

def main():
    # ==========================================
    # 1. CẤU HÌNH THÔNG SỐ
    # ==========================================
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    dataset_id = "simplescaling/s1K-1.1"
    output_base_name = "Distill_Qwen_32B_generated_outputs"
    num_splits = 5  # Tổng số lượng file zip 
    start_part = 3  # Bắt đầu chạy từ part 3 (Bỏ qua part 1 và 2)
    top_k_logprobs = 50 
    
    # ==========================================
    # 2. KHỞI TẠO MÔ HÌNH (vLLM)
    # ==========================================
    print(f"Đang tải mô hình {model_id}...")
    llm = LLM(
        model=model_id,
        gpu_memory_utilization=0.9,  
        tensor_parallel_size=4,  
        max_model_len=32768,
        language_model_only=True,
        max_logprobs=top_k_logprobs,
    )
    
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=32768,
        repetition_penalty=1.0,
        logprobs=top_k_logprobs
    )

    # ==========================================
    # 3. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
    # ==========================================
    print(f"Đang tải dữ liệu từ {dataset_id}...")
    dataset = load_dataset(dataset_id, split="train")
    
    prompts = []
    original_records = []

    print("Đang xử lý dữ liệu đầu vào...")
    
    system_prompt_text = (
        "You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
        "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
    )

    for item in dataset:
        messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": item["question"]}
        ]
        
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        prompts.append(prompt_text)
        original_records.append(item)

    # Tính toán vị trí cắt dữ liệu
    total_records = len(original_records)
    chunk_size = math.ceil(total_records / num_splits)
    global_start_idx = (start_part - 1) * chunk_size # Bỏ qua dữ liệu của các part trước

    # Chỉ lấy các prompt từ part 3 trở đi
    prompts_to_run = prompts[global_start_idx:]

    # ==========================================
    # 4. CHẠY SUY LUẬN (CHỈ CHO PART 3, 4, 5)
    # ==========================================
    print(f"Bắt đầu generate {len(prompts_to_run)} mẫu (từ part {start_part} đến {num_splits}). Quá trình này có thể mất một lúc...")
    outputs = llm.generate(prompts_to_run, sampling_params)

    # ==========================================
    # 5. LƯU KẾT QUẢ TENSOR VÀ TEXT RA FILE ZIP
    # ==========================================
    print(f"Đang lưu kết quả ra các file zip từ part {start_part} đến {num_splits}...")
    output_dir = "deepseek_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Chỉ lặp qua các part còn lại (VD: i = 2, 3, 4 tương ứng part 3, 4, 5)
    for i in range(start_part - 1, num_splits):
        # Tính toán vị trí toàn cục trong dataset gốc
        chunk_global_start = i * chunk_size
        chunk_global_end = min((i + 1) * chunk_size, total_records)
        
        if chunk_global_start >= total_records:
            break
            
        # Vị trí tương đối trong mảng `outputs` (vì outputs đã bị cắt bớt ở đầu)
        local_start = chunk_global_start - global_start_idx
        local_end = chunk_global_end - global_start_idx
        
        chunk_original = original_records[chunk_global_start:chunk_global_end]
        chunk_outputs = outputs[local_start:local_end]
        
        jsonl_filename = f"{output_base_name}_part_{i+1}.jsonl"
        tensor_filename = f"{output_base_name}_tensors_part_{i+1}.pt"
        zip_filename = f"{output_dir}/{output_base_name}_part_{i+1}.zip"
        
        tensor_map = {}

        with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            
            # 5.1 Ghi file JSONL
            with zf.open(jsonl_filename, 'w') as f:
                for idx_in_chunk, (original_data, output) in enumerate(zip(chunk_original, chunk_outputs)):
                    # Vẫn giữ nguyên global_idx để khớp với part 1 và 2
                    global_idx = chunk_global_start + idx_in_chunk
                    generated_text = output.outputs[0].text
                    
                    record = {
                        "global_idx": global_idx,
                        "question": original_data["question"],
                        "generated_response": generated_text.strip(),
                        "solution": original_data["solution"],
                    }
                    
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    f.write(line.encode('utf-8'))
                    
                    # 5.2 Xử lý Logprobs
                    step_token_ids = []
                    step_logprobs = []
                    
                    output_logprobs = output.outputs[0].logprobs
                    if output_logprobs is not None:
                        for step_probs_dict in output_logprobs:
                            sorted_probs = sorted(
                                step_probs_dict.items(), 
                                key=lambda x: x[1].logprob, 
                                reverse=True
                            )
                            top_k = sorted_probs[:top_k_logprobs]
                            
                            ids = [x[0] for x in top_k]
                            lps = [x[1].logprob for x in top_k]
                            
                            while len(ids) < top_k_logprobs:
                                ids.append(-1)
                                lps.append(float('-inf'))
                                
                            step_token_ids.append(ids)
                            step_logprobs.append(lps)
                    
                    tensor_map[global_idx] = {
                        "top_token_ids": torch.tensor(step_token_ids, dtype=torch.long),
                        "top_logprobs": torch.tensor(step_logprobs, dtype=torch.float16)
                    }

            # 5.3 Ghi Tensor map 
            buffer = io.BytesIO()
            torch.save(tensor_map, buffer)
            zf.writestr(tensor_filename, buffer.getvalue())
                    
        print(f"Đã lưu {zip_filename} ({len(chunk_original)} mẫu).")

    print("Hoàn thành việc generate lại các part còn thiếu!")

if __name__ == "__main__":
    main()