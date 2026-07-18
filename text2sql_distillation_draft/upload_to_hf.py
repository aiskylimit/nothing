from huggingface_hub import HfApi

HfApi().upload_folder(
    folder_path="results/",
    repo_id="distillation-sql/bird_baselines",
    repo_type="model",
)
