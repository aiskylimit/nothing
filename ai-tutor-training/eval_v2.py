import json
import os
import gc
import atexit
import hydra
import warnings
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from dotenv import load_dotenv

from src.classroom import Classroom, JudgeDecision
from utils.data import load_datasets
from config.eval import EvalConfig
from src.utils.utils import init_logger

load_dotenv()
logger = init_logger()
cs = ConfigStore.instance()
cs.store(name="config", node=EvalConfig)
warnings.filterwarnings("ignore")


def cleanup_classroom(classroom: Classroom | None) -> None:
    if classroom is None:
        return
    if getattr(classroom, "_eval_cleanup_done", False):
        return
    setattr(classroom, "_eval_cleanup_done", True)

    for model_name in ("teacher_model", "student_model", "judge_model", "reward_model"):
        model = getattr(classroom, model_name, None)
        cleanup = getattr(model, "cleanup", None)
        if cleanup is None:
            continue
        try:
            cleanup()
        except Exception as exc:
            logger.warning(f"Failed to cleanup {model_name}: {exc}")

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning(f"Failed to clear CUDA cache: {exc}")


@hydra.main(config_path="config/eval", version_base=None)
def main(cfg: EvalConfig):
    # Merge loaded config with defaults
    default_config = OmegaConf.structured(EvalConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    logger.info("Loading evaluation data and constructing the Classroom instance...")

    # Instantiate the Classroom for evaluation
    classroom = Classroom(
        cfg.student_model,
        cfg.teacher_model,
        cfg.judge_model,
        cfg.reward_model,
        cfg.generation,
        None,
    )
    atexit.register(cleanup_classroom, classroom)

    # Load evaluation datasets
    _, eval_data = load_datasets(cfg.dataset, cfg.seed)
    print(eval_data)

    _problems_we_sample = eval_data["problem"]
    _answers_we_sample = eval_data["answer"]

    number_of_times_to_average = cfg.num_samples_per_problem

    problem_we_sample = []
    answer_we_sample = []
    for i in range(len(_problems_we_sample)):
        problem_we_sample.extend([_problems_we_sample[i]] * number_of_times_to_average)
        answer_we_sample.extend([_answers_we_sample[i]] * number_of_times_to_average)

    logger.info("Sampling conversations...")
    conversations = classroom.sample_conversations(
        problem_we_sample,
        answer_we_sample,
        compute_initial_attempt=cfg.recompute_initial_attempts,
    )

    logger.info("Computing metrics...")

    if cfg.recompute_initial_attempts:
        # Compute reward deltas across conversations
        deltas = []
        for i in range(len(_problems_we_sample)):
            current_deltas = []
            for j in range(number_of_times_to_average):
                current_deltas.append(
                    conversations[
                        i * number_of_times_to_average + j
                    ].get_end_rm_reward()
                    - conversations[
                        i * number_of_times_to_average + j
                    ].get_initial_rm_reward()
                )
            deltas.append(sum(current_deltas) / len(current_deltas))
        delta_mean = sum(deltas) / len(deltas)
        print(f"Delta mean: {delta_mean}")

        # Mean before
        initial_rm_rewards = []
        for i in range(len(_problems_we_sample)):
            current_rewards = []
            for j in range(number_of_times_to_average):
                current_rewards.append(
                    conversations[
                        i * number_of_times_to_average + j
                    ].get_initial_rm_reward()
                )
            initial_rm_rewards.append(sum(current_rewards) / len(current_rewards))
        initial_rm_mean = sum(initial_rm_rewards) / len(initial_rm_rewards)
        print(f"Initial RM mean: {initial_rm_mean}")

    # Mean after
    end_rm_rewards = []
    for i in range(len(_problems_we_sample)):
        current_rewards = []
        for j in range(number_of_times_to_average):
            current_rewards.append(
                conversations[i * number_of_times_to_average + j].get_end_rm_reward()
            )
        end_rm_rewards.append(sum(current_rewards) / len(current_rewards))
    end_rm_mean = sum(end_rm_rewards) / len(end_rm_rewards)
    print(f"End RM mean: {end_rm_mean}")

    # Danh sách các tiêu chí đánh giá mới
    judge_criteria = [
        "accuracy_safety",
        "empathy_encouragement",
        "hint_quality",
        "leak_answer",
        "misconception_diagnosis",
        "style",
        "follows_pedagogical_values",
        "does_not_leak_answer"
    ]

    judge_metrics = {}

    print("--- Judge Metrics ---")
    for criterion in judge_criteria:
        criterion_reject_rates = []
        criterion_scores = []

        for i in range(len(_problems_we_sample)):
            current_reject_rates = []
            current_mean_scores = []

            for j in range(number_of_times_to_average):
                # Lấy danh sách JudgeResponse cho cuộc hội thoại hiện tại
                responses = conversations[
                    i * number_of_times_to_average + j
                ].judge_decisions.get(criterion, [])

                if not responses:
                    continue  # Bỏ qua nếu không có đánh giá nào để tránh lỗi chia cho 0

                # Tính tỉ lệ Reject
                reject_count = sum(1 for r in responses if r.decision == JudgeDecision.REJECT)
                current_reject_rates.append(reject_count / len(responses))

                # Tính điểm trung bình (score)
                total_score = sum(r.score for r in responses)
                current_mean_scores.append(total_score / len(responses))

            # Trung bình cho problem hiện tại
            if current_reject_rates:
                criterion_reject_rates.append(sum(current_reject_rates) / len(current_reject_rates))
            if current_mean_scores:
                criterion_scores.append(sum(current_mean_scores) / len(current_mean_scores))

        # Tính trung bình tổng thể cho toàn bộ các problems
        final_reject_mean = (sum(criterion_reject_rates) / len(criterion_reject_rates)) if criterion_reject_rates else 0.0
        final_score_mean = (sum(criterion_scores) / len(criterion_scores)) if criterion_scores else 0.0

        # Lưu vào dictionary với tên rõ ràng
        judge_metrics[f"{criterion}_reject_rate"] = final_reject_mean
        judge_metrics[f"{criterion}_mean_score"] = final_score_mean

        print(f"[{criterion}] Reject Rate: {final_reject_mean:.4f} | Mean Score: {final_score_mean:.4f}")

    print("---------------------\n")

    df_table = classroom.to_pd_latest()

    try:
        log = {
            "delta_mean": delta_mean if cfg.recompute_initial_attempts else 0,
            "initial_rm_rewards_mean": (
                initial_rm_mean if cfg.recompute_initial_attempts else 0
            ),
            "end_rm_rewards_mean": end_rm_mean,
        }

        rewards = [classroom.get_end_rm_reward(c) for c in conversations]
        df_table["end_rm_reward"] = rewards
        rewards = [classroom.get_thinking_reward(c) for c in conversations]
        df_table["thinking_reward"] = rewards
        rewards = [
            classroom.get_end_of_conversation_reward(c) for c in conversations
        ]
        df_table["end_of_conversation_reward"] = rewards
        rewards = [classroom.get_length_reward(c) for c in conversations]
        df_table["length_reward"] = rewards

        # sum of all rewards
        df_table["total_reward"] = (
            df_table["end_rm_reward"]
            + df_table["thinking_reward"]
            + df_table["end_of_conversation_reward"]
            + df_table["length_reward"]
        )
        df_table = df_table.astype(str)

        os.makedirs("eval_results", exist_ok=True)
        df_table.to_csv("eval_results/eval_results.csv", index=False)
        with open("eval_results/log.json", "w") as f:
            json.dump(log, f, indent=4)

    finally:
        cleanup_classroom(classroom)


if __name__ == "__main__":
    main()
