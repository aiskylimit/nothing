source ~/miniconda3/etc/profile.d/conda.sh
conda create -n vlm python=3.11 -y
conda activate vlm

rm -rf ./vlm
python -m venv vlm 
source vlm/bin/activate
pip install -r requirements.txt

#bash download_datatrain.sh

export CUDA_VISIBLE_DEVICES=4,5

bash script_train/run_baseline.sh
