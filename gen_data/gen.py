import os
import json
import zipfile
import math
from datasets import load_dataset
from vllm import LLM, SamplingParams

def main():
    # ==========================================
    # 1. CẤU HÌNH THÔNG SỐ
    # ==========================================
    model_id = "Qwen/Qwen3.6-27B-FP8"
    dataset_id = "VoCuc/AceReason-10k-SFT"
    output_base_name = "qwen_generated_outputs"
    num_splits = 5  # Số lượng file zip muốn chia nhỏ
    
    # ==========================================
    # 2. KHỞI TẠO MÔ HÌNH (vLLM)
    # ==========================================
    print(f"Đang tải mô hình {model_id}...")
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,  
        tensor_parallel_size=4,  
        max_model_len=10240
    )
    
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        max_tokens=10240,
        repetition_penalty=1.0
    )

    # ==========================================
    # 3. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
    # ==========================================
    print(f"Đang tải dữ liệu từ {dataset_id}...")
    dataset = load_dataset(dataset_id, split="train")
    dataset = dataset.shuffle(seed=42).select(range(10000))
    
    prompts = []
    original_records = []

    print("Đang xử lý dữ liệu đầu vào...")
    
    # Định nghĩa System Prompt (dùng raw string r"..." để tránh lỗi escape character \)
    system_prompt_text = r"Please enclose your final answer inside \box{}."

    for item in dataset:
        if "input" in item:
            # Xây dựng cấu trúc tin nhắn bao gồm System Prompt và User Input
            messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": item["input"]}
            ]
            
            # Áp dụng chat template của Qwen, tự động thêm prompt cho assistant bắt đầu gen
            prompt_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            prompts.append(prompt_text)
            original_records.append(item)
        else:
            continue

    # ==========================================
    # 4. CHẠY SUY LUẬN (BATCH GENERATION)
    # ==========================================
    print(f"Bắt đầu generate {len(prompts)} mẫu. Quá trình này có thể mất một lúc...")
    outputs = llm.generate(prompts, sampling_params)

    # ==========================================
    # 5. CHIA NHỎ VÀ LƯU KẾT QUẢ RA FILE ZIP
    # ==========================================
    print(f"Đang chia nhỏ và lưu kết quả ra {num_splits} file zip...")
    total_records = len(original_records)
    chunk_size = math.ceil(total_records / num_splits)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_records)
        
        # Tránh trường hợp mảng trống ở những phần dư cuối
        if start_idx >= total_records:
            break
            
        chunk_original = original_records[start_idx:end_idx]
        chunk_outputs = outputs[start_idx:end_idx]
        
        jsonl_filename = f"{output_base_name}_part_{i+1}.jsonl"
        zip_filename = f"{output_dir}/{output_base_name}_part_{i+1}.zip"
        
        # Mở file nén (sử dụng ZIP_DEFLATED để nén giảm dung lượng)
        with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # Mở một luồng ghi vào file jsonl nội bộ bên trong zip
            with zf.open(jsonl_filename, 'w') as f:
                for original_data, output in zip(chunk_original, chunk_outputs):
                    generated_text = output.outputs[0].text
                    
                    record = {
                        "dataset_input": original_data["input"],
                        "qwen_generated_response": generated_text.strip()
                    }
                    
                    # Encode dạng utf-8 dạng byte để ghi vào zip
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    f.write(line.encode('utf-8'))
                    
        print(f"Đã lưu {zip_filename} ({end_idx - start_idx} mẫu).")

    print("Hoàn thành! Đã lưu toàn bộ kết quả dưới dạng các file zip riêng biệt.")

if __name__ == "__main__":
    main()