from __future__ import annotations

from dataclasses import dataclass
import string
from typing import Optional
import re
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


SQL_KEYWORDS = {
    "ADD",
    "ALL",
    "ALTER",
    "AND",
    "AS",
    "ASC",
    "BETWEEN",
    "BY",
    "CASE",
    "COUNT",
    "CREATE",
    "DELETE",
    "DESC",
    "DISTINCT",
    "DROP",
    "ELSE",
    "END",
    "EXCEPT",
    "EXISTS",
    "FROM",
    "GROUP",
    "HAVING",
    "IN",
    "INNER",
    "INSERT",
    "INTERSECT",
    "INTO",
    "IS",
    "JOIN",
    "LEFT",
    "LIKE",
    "LIMIT",
    "MAX",
    "MIN",
    "NOT",
    "NULL",
    "ON",
    "OR",
    "ORDER",
    "OUTER",
    "RIGHT",
    "SELECT",
    "SET",
    "SUM",
    "THEN",
    "UNION",
    "UPDATE",
    "VALUES",
    "WHEN",
    "WHERE",
    "WITH",
}

_TOKEN_TEXT_CACHES: WeakKeyDictionary = WeakKeyDictionary()


@dataclass
class SynIDLossParts:
    total: torch.Tensor
    kd: torch.Tensor
    con1: torch.Tensor
    con2: torch.Tensor


def combine_synid_with_ce(
    ce_loss: torch.Tensor,
    loss_parts: SynIDLossParts,
    *,
    kd_ratio: float,
    con1_weight: float,
    con2_weight: float,
) -> torch.Tensor:
    if not 0.0 <= kd_ratio <= 1.0:
        raise ValueError(f"SynID KD ratio must be in [0, 1], got {kd_ratio}.")
    if con1_weight < 0 or con2_weight < 0:
        raise ValueError("SynID contrastive loss weights must be non-negative.")
    return (
        (1.0 - kd_ratio) * ce_loss
        + kd_ratio
        * (
            loss_parts.kd
            + con1_weight * loss_parts.con1
            + con2_weight * loss_parts.con2
        )
    )


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _compact_by_label_mask(values: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
    mask = labels != -100
    return [values[i, mask[i]] for i in range(values.size(0))]


def _token_forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Token-wise equivalent of ``distillm.losses.forward_kl``."""
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(student_logits)
    student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_log_probs, inf_mask, 0)
    return -torch.sum(prod_probs, dim=-1)


def _token_reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Token-wise equivalent of ``distillm.losses.reverse_kl``."""
    student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32)
    student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
    prod_probs = torch.masked_fill(student_probs * teacher_log_probs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_log_probs, inf_mask, 0)
    return -torch.sum(prod_probs, dim=-1)


def _token_skewed_forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, lam: float) -> torch.Tensor:
    """Token-wise equivalent of ``distillm.losses.skewed_forward_kl``."""
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1.0 - lam) * student_probs
    mixed_log_probs = torch.log(mixed_probs)
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
    prod_probs = torch.masked_fill(teacher_probs * mixed_log_probs, inf_mask, 0)
    return -torch.sum(prod_probs, dim=-1)


def _token_skewed_reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, lam: float) -> torch.Tensor:
    """Token-wise equivalent of ``distillm.losses.skewed_reverse_kl``."""
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1.0 - lam) * teacher_probs + lam * student_probs
    student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    mixed_log_probs = torch.log(mixed_probs)
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
    prod_probs = torch.masked_fill(student_probs * mixed_log_probs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_log_probs, inf_mask, 0)
    return -torch.sum(prod_probs, dim=-1)


def _token_csd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """CSD-SS surrogate loss, matching ``distillm.losses.csd`` token-wise."""
    student_probs = F.softmax(student_logits / temperature, dim=-1, dtype=torch.float32)
    logit_gap = student_logits - teacher_logits
    centered_gap = logit_gap - (student_probs * logit_gap).sum(dim=-1, keepdim=True)
    surrogate = centered_gap.detach() * student_probs.detach() * student_logits
    return surrogate.sum(dim=-1)


def response_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_labels: Optional[torch.Tensor] = None,
    kd_loss: str = "fkl",
    skew_alpha: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KD divergence over response positions, compacted per sample.

    Student and teacher prompts may have different lengths in later phases, so
    this aligns response tokens by the label masks instead of absolute positions.
    """
    if temperature <= 0:
        raise ValueError(f"SynID KD temperature must be positive, got {temperature}.")
    if kd_loss in {"sfkl", "srkl"} and not 0.0 <= skew_alpha <= 1.0:
        raise ValueError(f"SynID skew alpha must be in [0, 1], got {skew_alpha}.")
    if student_logits.size(0) != student_labels.size(0):
        raise ValueError("Student logits and labels must have the same batch size.")

    if teacher_labels is None:
        teacher_labels = student_labels
    if teacher_logits.size(0) != teacher_labels.size(0):
        raise ValueError("Teacher logits and labels must have the same batch size.")
    if student_logits.size(0) != teacher_logits.size(0):
        raise ValueError("Student and teacher batches must have the same size.")
    if student_logits.size(-1) != teacher_logits.size(-1):
        raise ValueError("Student and teacher logits must use the same vocabulary.")

    student_parts = _compact_by_label_mask(student_logits, student_labels)
    teacher_parts = _compact_by_label_mask(teacher_logits, teacher_labels)
    student_token_parts = _compact_by_label_mask(student_labels, student_labels)
    teacher_token_parts = _compact_by_label_mask(teacher_labels, teacher_labels)

    loss_sum = student_logits.new_zeros((), dtype=torch.float32)
    token_count = 0
    for row, (s_logits, t_logits, s_tokens, t_tokens) in enumerate(
        zip(student_parts, teacher_parts, student_token_parts, teacher_token_parts)
    ):
        if s_logits.size(0) != t_logits.size(0):
            raise ValueError(
                "Student and teacher response lengths differ at batch row "
                f"{row}: {s_logits.size(0)} != {t_logits.size(0)}."
            )
        if s_logits.size(0) == 0:
            raise ValueError(f"SynID received an empty response at batch row {row}.")
        if not torch.equal(s_tokens, t_tokens):
            raise ValueError(f"Student and teacher response tokens differ at batch row {row}.")
        if kd_loss == "fkl":
            s_logits = s_logits.float() / temperature
            t_logits = t_logits.float() / temperature
            token_loss = _token_forward_kl(s_logits, t_logits)
        elif kd_loss == "rkl":
            s_logits = s_logits.float() / temperature
            t_logits = t_logits.float() / temperature
            token_loss = _token_reverse_kl(s_logits, t_logits)
        elif kd_loss == "sfkl":
            s_logits = s_logits.float() / temperature
            t_logits = t_logits.float() / temperature
            token_loss = _token_skewed_forward_kl(s_logits, t_logits, skew_alpha)
        elif kd_loss == "srkl":
            s_logits = s_logits.float() / temperature
            t_logits = t_logits.float() / temperature
            token_loss = _token_skewed_reverse_kl(s_logits, t_logits, skew_alpha)
        elif kd_loss == "csd":
            token_loss = _token_csd(s_logits.float(), t_logits.float(), temperature)
        else:
            raise ValueError(f"Unsupported SynID KD loss: {kd_loss}")
        if kd_loss == "csd":
            loss_sum = loss_sum + token_loss.sum()
        else:
            loss_sum = loss_sum + token_loss.sum() * (temperature ** 2)
        token_count += int(token_loss.numel())

    return loss_sum / token_count


def _last_valid_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
    last_positions = positions.masked_fill(~mask, -1).max(dim=1).values.clamp(min=0)
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, last_positions]


def _normalize_token_text(text: str) -> str:
    return text.strip().strip(string.punctuation).upper()


def _term_variants(term: str) -> set[str]:
    normalized = _normalize_token_text(term)
    if not normalized:
        return set()
    variants = {normalized}
    variants.update(
        part
        for part in re.split(r"[_\W]+", normalized)
        if len(part) >= 4 and part not in SQL_KEYWORDS
    )
    return variants


def _extract_schema_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- ") or "(" not in line or ")" not in line:
            continue
        table_part, columns_part = line[2:].split("(", 1)
        terms.update(_term_variants(table_part))
        columns = columns_part.split(")", 1)[0]
        for column in columns.split(","):
            terms.update(_term_variants(column))
    return terms


def _normalized_token_texts(tokenizer, token_ids: set[int]) -> dict[int, str]:
    try:
        cache = _TOKEN_TEXT_CACHES.setdefault(tokenizer, {})
    except TypeError:
        cache = getattr(tokenizer, "_synid_normalized_token_cache", None)
        if cache is None:
            cache = {}
            setattr(tokenizer, "_synid_normalized_token_cache", cache)

    missing = sorted(token_ids.difference(cache))
    if missing:
        if hasattr(tokenizer, "batch_decode"):
            decoded = tokenizer.batch_decode(
                [[token_id] for token_id in missing],
                clean_up_tokenization_spaces=False,
            )
        else:
            decoded = [
                tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                for token_id in missing
            ]
        cache.update(
            (token_id, _normalize_token_text(text))
            for token_id, text in zip(missing, decoded)
        )
    return {token_id: cache[token_id] for token_id in token_ids}


def _syntax_weights(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    tokenizer,
    syntax_lambda: float,
    use_syntax_weights: bool,
    context_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    weights = torch.ones_like(mask, dtype=torch.float32)
    if not use_syntax_weights:
        return weights

    ids_cpu = input_ids.detach().cpu()
    mask_cpu = mask.detach().cpu()
    context_mask_cpu = context_mask.detach().cpu() if context_mask is not None else mask_cpu

    token_ids = {
        int(ids_cpu[row, col])
        for row in range(ids_cpu.size(0))
        for col in range(ids_cpu.size(1))
        if mask_cpu[row, col].item()
    }
    normalized_tokens = _normalized_token_texts(tokenizer, token_ids)

    for row in range(ids_cpu.size(0)):
        valid_ids = [int(ids_cpu[row, col]) for col in range(ids_cpu.size(1)) if context_mask_cpu[row, col].item()]
        row_text = tokenizer.decode(valid_ids, clean_up_tokenization_spaces=False)
        schema_terms = _extract_schema_terms(row_text)
        for col in range(ids_cpu.size(1)):
            if not mask_cpu[row, col].item():
                continue
            token_id = int(ids_cpu[row, col])
            normalized = normalized_tokens[token_id]
            if normalized in SQL_KEYWORDS or normalized in schema_terms:
                weights[row, col] = syntax_lambda

    return weights.to(device=input_ids.device, dtype=torch.float32)


def _pool_hidden(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    tokenizer,
    pooling: str,
    pool_tau: float,
    syntax_lambda: float,
    use_syntax_weights: bool,
    context_mask: Optional[torch.Tensor] = None,
    syntax_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = mask.bool()
    if pooling == "mean":
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return (hidden_states * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    if pooling != "sc":
        raise ValueError(f"Unsupported SynID pooling mode: {pooling}")
    if pool_tau <= 0:
        raise ValueError(f"SynID pooling temperature must be positive, got {pool_tau}.")

    valid_rows = mask.any(dim=1)
    safe_mask = mask.clone()
    safe_mask[~valid_rows, 0] = True
    anchor = _last_valid_hidden(hidden_states, safe_mask)
    cosine = F.cosine_similarity(hidden_states.float(), anchor[:, None, :].float(), dim=-1)
    if syntax_weights is None:
        syntax_weights = _syntax_weights(
            input_ids,
            safe_mask,
            tokenizer,
            syntax_lambda,
            use_syntax_weights,
            context_mask,
        )
    scores = cosine * syntax_weights / pool_tau
    scores = scores.masked_fill(~safe_mask, torch.finfo(scores.dtype).min)
    alpha = torch.softmax(scores, dim=1).to(hidden_states.dtype)
    pooled = (hidden_states * alpha.unsqueeze(-1)).sum(dim=1)
    return pooled.masked_fill(~valid_rows.unsqueeze(-1), 0)


def _response_hidden_mask(labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    label_mask = labels != -100
    hidden_mask = torch.zeros_like(label_mask, dtype=torch.bool)
    hidden_mask[:, 1:] = label_mask[:, :-1]
    return hidden_mask & attention_mask.bool()


def _all_gather_variable_batch_no_grad(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not dist.is_available() or not dist.is_initialized():
        return values, 0

    world_size = dist.get_world_size()
    if world_size <= 1:
        return values, 0

    rank = dist.get_rank()
    local_size = torch.tensor([values.size(0)], device=values.device, dtype=torch.long)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    max_size = max(sizes)
    if max_size == 0:
        return values[:0], 0

    if values.size(0) < max_size:
        pad_shape = (max_size - values.size(0), *values.shape[1:])
        padding = values.new_zeros(pad_shape)
        padded = torch.cat([values, padding], dim=0)
    else:
        padded = values

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded.contiguous())
    global_values = torch.cat(
        [rank_values[:size] for rank_values, size in zip(gathered, sizes)],
        dim=0,
    )
    label_offset = sum(sizes[:rank])
    return global_values, label_offset


def _info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"SynID contrastive temperature must be positive, got {temperature}.")
    if anchor.shape != positive.shape:
        raise ValueError(
            f"SynID contrastive embeddings must have equal shapes, got {anchor.shape} and {positive.shape}."
        )
    anchor = F.normalize(anchor.float(), dim=-1)
    positive = F.normalize(positive.float(), dim=-1)
    with torch.no_grad():
        global_positive, label_offset = _all_gather_variable_batch_no_grad(positive.detach())
    if global_positive.size(0) <= 1:
        return _zero_like(anchor)
    logits = anchor @ global_positive.t()
    logits = logits / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device) + label_offset
    return F.cross_entropy(logits, labels)


def _project_if_needed(embeddings: torch.Tensor, projector: Optional[nn.Module]) -> torch.Tensor:
    if projector is None:
        return embeddings
    return projector(embeddings)


def _as_hidden_state_list(hidden_states) -> list[torch.Tensor]:
    if isinstance(hidden_states, torch.Tensor):
        return [hidden_states]
    if isinstance(hidden_states, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in hidden_states):
        if len(hidden_states) == 0:
            raise ValueError("SynID received an empty hidden-state layer list.")
        return list(hidden_states)
    raise TypeError("SynID hidden states must be a tensor or a non-empty list of tensors.")


def synid_loss(
    *,
    args,
    tokenizer,
    student_outputs,
    teacher_outputs,
    student_batch: dict[str, torch.Tensor],
    student_no_model_batch: dict[str, torch.Tensor],
    teacher_batch: Optional[dict[str, torch.Tensor]] = None,
    teacher_no_model_batch: Optional[dict[str, torch.Tensor]] = None,
    student_projector: Optional[nn.Module] = None,
    student_hidden_states: Optional[torch.Tensor] = None,
    teacher_hidden_states: Optional[torch.Tensor] = None,
) -> SynIDLossParts:
    if teacher_batch is None:
        teacher_batch = student_batch
    if teacher_no_model_batch is None:
        teacher_no_model_batch = student_no_model_batch

    student_labels = student_no_model_batch["label"]
    teacher_labels = teacher_no_model_batch["label"]

    kd_loss = response_kd_loss(
        student_outputs.logits,
        teacher_outputs.logits,
        student_labels,
        teacher_labels,
        kd_loss=args.synid_kd_loss,
        skew_alpha=args.skew_alpha,
        temperature=args.synid_kd_temperature,
    )

    if args.synid_pool_tau <= 0:
        raise ValueError(f"SynID pooling temperature must be positive, got {args.synid_pool_tau}.")
    if args.synid_syntax_lambda <= 0:
        raise ValueError(f"SynID syntax lambda must be positive, got {args.synid_syntax_lambda}.")
    if args.synid_alpha < 0 or args.synid_beta < 0:
        raise ValueError("SynID contrastive loss weights must be non-negative.")

    if student_hidden_states is None:
        student_hidden_states = student_outputs.hidden_states[-1]
    if teacher_hidden_states is None:
        teacher_hidden_states = teacher_outputs.hidden_states[-1]
    student_hidden_layers = _as_hidden_state_list(student_hidden_states)
    teacher_hidden_layers = _as_hidden_state_list(teacher_hidden_states)
    if len(student_hidden_layers) != len(teacher_hidden_layers):
        raise ValueError(
            "Student and teacher hidden-state layer lists must have equal length, "
            f"got {len(student_hidden_layers)} and {len(teacher_hidden_layers)}."
        )

    student_attention_mask = student_batch["attention_mask"].bool()
    teacher_attention_mask = teacher_batch["attention_mask"].bool()
    student_response_mask = _response_hidden_mask(student_labels, student_attention_mask)
    teacher_response_mask = _response_hidden_mask(teacher_labels, teacher_attention_mask)
    student_prompt_mask = student_attention_mask & ~student_response_mask

    student_syntax_weights = None
    teacher_syntax_weights = None
    if args.synid_pooling == "sc":
        student_syntax_weights = _syntax_weights(
            student_batch["input_ids"],
            student_attention_mask,
            tokenizer,
            args.synid_syntax_lambda,
            args.synid_use_syntax_weights,
            student_attention_mask,
        )
        if teacher_batch is student_batch:
            teacher_syntax_weights = student_syntax_weights
        else:
            teacher_syntax_weights = _syntax_weights(
                teacher_batch["input_ids"],
                teacher_attention_mask,
                tokenizer,
                args.synid_syntax_lambda,
                args.synid_use_syntax_weights,
                teacher_attention_mask,
            )

    con1_losses = []
    con2_losses = []
    for student_hidden, teacher_hidden in zip(student_hidden_layers, teacher_hidden_layers):
        # Teacher response is the positive solution representation. In phase 1 it is
        # from the same prompt as the student; later it can come from privileged input.
        teacher_response_embedding = _pool_hidden(
            teacher_hidden,
            teacher_batch["input_ids"],
            teacher_response_mask,
            tokenizer,
            args.synid_pooling,
            args.synid_pool_tau,
            args.synid_syntax_lambda,
            args.synid_use_syntax_weights,
            teacher_attention_mask,
            teacher_syntax_weights,
        )
        student_prompt_embedding = _pool_hidden(
            student_hidden,
            student_batch["input_ids"],
            student_prompt_mask,
            tokenizer,
            args.synid_pooling,
            args.synid_pool_tau,
            args.synid_syntax_lambda,
            args.synid_use_syntax_weights,
            student_attention_mask,
            student_syntax_weights,
        )
        student_response_embedding = _pool_hidden(
            student_hidden,
            student_batch["input_ids"],
            student_response_mask,
            tokenizer,
            args.synid_pooling,
            args.synid_pool_tau,
            args.synid_syntax_lambda,
            args.synid_use_syntax_weights,
            student_attention_mask,
            student_syntax_weights,
        )

        student_prompt_embedding = _project_if_needed(student_prompt_embedding, student_projector)
        student_response_embedding = _project_if_needed(student_response_embedding, student_projector)

        con1_losses.append(
            _zero_like(kd_loss)
            if not args.synid_use_con1
            else _info_nce(student_prompt_embedding, teacher_response_embedding, args.synid_contrastive_tau)
        )
        con2_losses.append(
            _zero_like(kd_loss)
            if not args.synid_use_con2
            else _info_nce(student_response_embedding, teacher_response_embedding, args.synid_contrastive_tau)
        )

    con1_loss = torch.stack(con1_losses).mean()
    con2_loss = torch.stack(con2_losses).mean()

    total = kd_loss + args.synid_alpha * con1_loss + args.synid_beta * con2_loss
    return SynIDLossParts(total=total, kd=kd_loss, con1=con1_loss, con2=con2_loss)