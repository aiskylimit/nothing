import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen2.5-1.5B"
lora_model = "ermiaazarkhalili/Qwen2.5-1.5B-SFT-UltraChat"

save_path = "./Qwen2.5-1.5B-SFT-UltraChat-merged"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# load LoRA
model = PeftModel.from_pretrained(
    model,
    lora_model,
)

# merge LoRA vào base
model = model.merge_and_unload()

# lưu full model
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)

print("Saved to:", save_path)