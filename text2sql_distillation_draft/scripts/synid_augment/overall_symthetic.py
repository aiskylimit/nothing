import json
from pathlib import Path

def main():
    # 1. Khai báo các đường dẫn
    base_dir = Path("benchmarks_2/spider_data/synid_aug_v2_lora")
    final_rejected_path = base_dir / "rejected_final.jsonl"
    loops_dir = base_dir / "loops"
    recovered_path = base_dir / "recovered_final.jsonl" # File xuất ra cho các sample được cứu
    accepted_path = base_dir / "accepted_all.jsonl"     # File chứa các sample đã pass
    final_merged_path = base_dir / "final_merged.jsonl" # File gộp cuối cùng

    if not final_rejected_path.exists():
        print(f"Không tìm thấy file: {final_rejected_path}")
        return

    # 2. Đọc danh sách các sample bị fail cuối cùng
    final_rejected_records = {}
    with open(final_rejected_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            record_id = str(record["id"])
            final_rejected_records[record_id] = record

    print(f"Đã load {len(final_rejected_records)} samples từ rejected_final.jsonl")

    # 3. Quét tất cả các loop để tìm best candidate cho từng sample fail
    best_candidates = {}
    
    # Tìm tất cả các file rejected.jsonl trong các thư mục loop_1, loop_2, loop_3...
    for loop_file in loops_dir.glob("loop_*/rejected.jsonl"):
        with open(loop_file, "r", encoding="utf-8") as f:
            for line in f:
                cand = json.loads(line)
                cand_id = str(cand.get("id"))
                
                # Nếu sample này nằm trong danh sách fail cuối cùng
                if cand_id in final_rejected_records:
                    # Chỉ lấy nếu lý do fail là jaccard_too_high
                    if cand.get("reason") == "jaccard_too_high":
                        current_jaccard = cand.get("jaccard", 1.0)
                        
                        # Cập nhật nếu chưa có, hoặc nếu jaccard này nhỏ hơn jaccard đã lưu
                        if cand_id not in best_candidates:
                            best_candidates[cand_id] = cand
                        else:
                            best_jaccard = best_candidates[cand_id].get("jaccard", 1.0)
                            if current_jaccard < best_jaccard:
                                best_candidates[cand_id] = cand

    # 4. Quyết định kết quả cuối cùng (lấy từ loop hoặc fallback về gold)
    recovered_records = []
    stats = {"recovered_from_loops": 0, "fallback_to_gold": 0}

    for cand_id, original_record in final_rejected_records.items():
        recovered_record = original_record.copy()
        
        # NOTE: Thay đổi key "query" dưới đây thành tên key chứa câu SQL gốc của bạn
        gold_sql = original_record.get("query", original_record.get("gold_sql", ""))

        if cand_id in best_candidates:
            # Nếu tìm thấy candidate hợp lệ trong các loop
            best = best_candidates[cand_id]
            recovered_record["candidate_sql"] = best["candidate_sql"]
            recovered_record["jaccard"] = best["jaccard"]
            recovered_record["recovery_source"] = f"loop_{best.get('loop', 'unknown')}"
            stats["recovered_from_loops"] += 1
        else:
            # Không tìm thấy ứng viên hợp lệ -> Fallback về câu Gold SQL
            recovered_record["candidate_sql"] = gold_sql
            recovered_record["jaccard"] = 1.0 # Jaccard của gold vs gold là 1.0
            recovered_record["recovery_source"] = "gold_fallback"
            stats["fallback_to_gold"] += 1
            
        recovered_records.append(recovered_record)

    # 5. Lưu ra file recovered_final.jsonl (Bước đệm)
    with open(recovered_path, "w", encoding="utf-8") as f:
        for rec in recovered_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 6. Đọc file accepted_all.jsonl và gộp (Merge)
    accepted_records = []
    if accepted_path.exists():
        with open(accepted_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                # Đánh dấu nguồn gốc cho đồng nhất
                rec["recovery_source"] = "accepted"
                accepted_records.append(rec)
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy file {accepted_path}. Sẽ chỉ xuất ra dữ liệu recovered.")

    # Gộp 2 mảng lại với nhau
    final_merged_records = accepted_records + recovered_records

    # Tùy chọn: Sắp xếp lại dữ liệu theo 'id' cho dễ nhìn (nếu id là số)
    try:
        final_merged_records.sort(key=lambda x: int(x["id"]))
    except ValueError:
        # Nếu id chứa chữ (không phải số nguyên), sắp xếp theo kiểu chuỗi
        final_merged_records.sort(key=lambda x: str(x["id"]))

    # Lưu ra file gộp cuối cùng
    with open(final_merged_path, "w", encoding="utf-8") as f:
        for rec in final_merged_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # In báo cáo
    print("\n--- KẾT QUẢ KHÔI PHỤC & GỘP DATA ---")
    print(f"✅ Số sample pass ngay từ đầu (Accepted): {len(accepted_records)}")
    print(f"🔄 Số sample được khôi phục/fallback: {len(recovered_records)}")
    print(f"   ↳ Khôi phục từ các loop (Min Jaccard): {stats['recovered_from_loops']}")
    print(f"   ↳ Dùng bản Gold SQL (Fallback): {stats['fallback_to_gold']}")
    print("-" * 35)
    print(f"🚀 TỔNG CỘNG data sau khi gộp: {len(final_merged_records)} samples")
    print(f"📁 Đã lưu file khôi phục tại: {recovered_path}")
    print(f"📁 Đã lưu file gộp toàn bộ tại: {final_merged_path}")

if __name__ == "__main__":
    main()