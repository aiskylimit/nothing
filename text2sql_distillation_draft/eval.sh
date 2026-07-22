INFER_OUTPUT_ROOT=results/infer/synid_ce_no_keywords_weight_lora_218/qwen_updated_2 \
EVAL_OUTPUT_ROOT=results/eval/synid_ce_no_keywords_weight_lora_218/qwen_updated_2 \
SKIP_EXISTING_EVAL=0 \
PROGRESS_BAR=1 \
bash scripts/qwen_updated_2/synid_ce_no_keywords_weight_lora_218/format_eval_multiseed.sh \
  spider_data spider_syn spider_realistic spider_dk


INFER_OUTPUT_ROOT=results/infer/synid_ce_no_keywords_weight_lora_436/qwen_updated \
EVAL_OUTPUT_ROOT=results/eval/synid_ce_no_keywords_weight_lora_436/qwen_updated \
SKIP_EXISTING_EVAL=0 \
PROGRESS_BAR=1 \
bash scripts/qwen_updated/synid_ce_no_keywords_weight_lora_436/format_eval_multiseed.sh \
  spider_data spider_syn spider_realistic spider_dk

INFER_OUTPUT_ROOT=results/infer/synid_ce_keywords_weight_lora_218/qwen_updated_2 \
EVAL_OUTPUT_ROOT=results/eval/synid_ce_keywords_weight_lora_218/qwen_updated_2 \
SKIP_EXISTING_EVAL=0 \
PROGRESS_BAR=1 \
bash scripts/qwen_updated_2/synid_ce_keywords_weight_lora_218/format_eval_multiseed.sh \
  spider_data spider_syn spider_realistic spider_dk


INFER_OUTPUT_ROOT=results/infer/synid_ce_keywords_weight_lora_436/qwen_updated \
EVAL_OUTPUT_ROOT=results/eval/synid_ce_keywords_weight_lora_436/qwen_updated \
SKIP_EXISTING_EVAL=0 \
PROGRESS_BAR=1 \
bash scripts/qwen_updated/synid_ce_keywords_weight_lora_436/format_eval_multiseed.sh \
  spider_data spider_syn spider_realistic spider_dk