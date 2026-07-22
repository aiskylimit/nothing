from huggingface_hub import HfApi


HfApi().upload_folder(
    folder_path="results/",
    repo_id="Dream-AI-HUST/text2sql-distillation-results",
    repo_type="model",
)
