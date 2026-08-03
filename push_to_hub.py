import os

from huggingface_hub import create_repo, upload_folder

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    raise ValueError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before uploading.")

folder_path = "training/SEGD_FastVLM_cls_r32_bs12_ckalast/checkpoint-final"
repo_id = "vohuutridung/SEGD_FastVLM_cls_r32_bs12_ckalast"

create_repo(repo_id=repo_id, token=token, exist_ok=True)
upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    token=token,
    path_in_repo="",
)

print("Uploaded folder to Hugging Face successfully!")
