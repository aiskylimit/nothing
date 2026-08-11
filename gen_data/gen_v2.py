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
    num_splits = 5  # Số lượng file zip muốn chia nhỏ
    top_k_logprobs = 50 # Số lượng top logprob muốn lấy
    
    # ==========================================
    # 2. KHỞI TẠO MÔ HÌNH (vLLM)
    # ==========================================
    print(f"Đang tải mô hình {model_id}...")
    llm = LLM(
        model=model_id,
        gpu_memory_utilization=0.9,  
        tensor_parallel_size=8,  
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
        logprobs=top_k_logprobs  # Bật tính năng lấy logprobs
    )

    # ==========================================
    # 3. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
    # ==========================================
    print(f"Đang tải dữ liệu từ {dataset_id}...")
    dataset = load_dataset(dataset_id, split="train")
    # dataset = dataset.shuffle(seed=42).select(range(10000))
    
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

    # ==========================================
    # 4. CHẠY SUY LUẬN (BATCH GENERATION)
    # ==========================================
    print(f"Bắt đầu generate {len(prompts)} mẫu. Quá trình này có thể mất một lúc...")
    outputs = llm.generate(prompts, sampling_params)

    # ==========================================
    # 5. CHIA NHỎ VÀ LƯU KẾT QUẢ TENSOR VÀ TEXT RA FILE ZIP
    # ==========================================
    print(f"Đang chia nhỏ và lưu kết quả (Text + Tensor) ra {num_splits} file zip...")
    total_records = len(original_records)
    chunk_size = math.ceil(total_records / num_splits)

    output_dir = "deepseek_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_records)
        
        if start_idx >= total_records:
            break
            
        chunk_original = original_records[start_idx:end_idx]
        chunk_outputs = outputs[start_idx:end_idx]
        
        jsonl_filename = f"{output_base_name}_part_{i+1}.jsonl"
        tensor_filename = f"{output_base_name}_tensors_part_{i+1}.pt"
        zip_filename = f"{output_dir}/{output_base_name}_part_{i+1}.zip"
        
        # Dictionary để lưu tensor map của chunk hiện tại
        tensor_map = {}

        with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            
            # 5.1 Xử lý và ghi file JSONL
            with zf.open(jsonl_filename, 'w') as f:
                for idx_in_chunk, (original_data, output) in enumerate(zip(chunk_original, chunk_outputs)):
                    global_idx = start_idx + idx_in_chunk
                    generated_text = output.outputs[0].text
                    
                    record = {
                        "global_idx": global_idx,
                        "question": original_data["question"],
                        "generated_response": generated_text.strip(),
                        "solution": original_data["solution"],
                    }
                    
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    f.write(line.encode('utf-8'))
                    
                    # 5.2 Xử lý Logprobs để đưa vào Tensor Map
                    step_token_ids = []
                    step_logprobs = []
                    
                    output_logprobs = output.outputs[0].logprobs
                    if output_logprobs is not None:
                        for step_probs_dict in output_logprobs:
                            # Sắp xếp để chắc chắn lấy từ cao xuống thấp
                            sorted_probs = sorted(
                                step_probs_dict.items(), 
                                key=lambda x: x[1].logprob, 
                                reverse=True
                            )
                            # Lấy top 50
                            top_k = sorted_probs[:top_k_logprobs]
                            
                            ids = [x[0] for x in top_k]
                            lps = [x[1].logprob for x in top_k]
                            
                            # Đảm bảo đủ chiều dài tensor bằng padding (nếu < 50)
                            while len(ids) < top_k_logprobs:
                                ids.append(-1)
                                lps.append(float('-inf'))
                                
                            step_token_ids.append(ids)
                            step_logprobs.append(lps)
                    
                    # Lưu tensor vào map với key là global_idx
                    # Shape: (L, 50) - lưu dtype=torch.float16 để tiết kiệm dung lượng
                    tensor_map[global_idx] = {
                        "top_token_ids": torch.tensor(step_token_ids, dtype=torch.long),
                        "top_logprobs": torch.tensor(step_logprobs, dtype=torch.float16) # <--- Sửa đổi ở đây
                    }

            # 5.3 Ghi Tensor map trực tiếp vào file ZIP 
            buffer = io.BytesIO()
            torch.save(tensor_map, buffer)
            zf.writestr(tensor_filename, buffer.getvalue())
                    
        print(f"Đã lưu {zip_filename} ({end_idx - start_idx} mẫu).")

    print("Hoàn thành! Đã lưu toàn bộ kết quả dưới dạng các file zip riêng biệt.")

if __name__ == "__main__":
    main()