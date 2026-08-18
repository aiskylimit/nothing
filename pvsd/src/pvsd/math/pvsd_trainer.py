"""PVSD trainer: privilege-vector self-distillation in activation space.

The teacher is the student itself, steered at one layer by a purified privilege
vector. Privileged text never enters the context of the forward pass that produces
the teacher distribution.

Per training step:

1. sample an on-policy rollout from the student (handled by ``AVSDTrainer``);
2. every ``pvsd_pie_every`` steps, localise a head set ``A^(m)`` per view with PIE;
3. read raw privilege vectors from prompt-only prefills over ``(x, r^(m))`` at the
   last input token, from those heads;
4. purify each view against its matched corrupted contexts ``(x, r~_k^(m))``;
5. fuse the purified view vectors into one vector ``v*``;
6. run the student again with ``h_l* <- h_l* + alpha * v*`` to get the teacher
   distribution ``q*`` (stop-gradient);
7. distil the student onto ``q*`` with reverse KL (``--beta 1.0``).
"""

from __future__ import annotations

import torch
from accelerate.utils import broadcast_object_list

try:
    from trl.trainer.utils import empty_cache
except ImportError:  # pragma: no cover - compatibility fallback for older TRL installs
    def empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

from pvsd.common.pie import (
    PIE_SCORE_MODES,
    compute_privilege_indirect_effect,
    estimate_pie_forward_passes,
)
from pvsd.common.privilege_vectors import (
    PURIFICATION_MODES,
    all_heads_in_layers,
    build_privilege_vectors_from_batch,
    cosine_similarity_rowwise,
    describe_model,
    head_set_jaccard,
    inject_at_layer,
    normalize_head_set,
    position_ids_from_mask,
)
from pvsd.common.token_log_probs import sampled_token_log_probs
from pvsd.math.privileged_views import VIEW_TYPES
from pvsd.math.trainer import AVSDTrainer


STEER_SCOPES = ("completion", "all")


class PVSDTrainer(AVSDTrainer):
    """AVSD trainer whose teacher signal is an injected privilege vector."""

    pvsd_enabled = True

    def __init__(
        self,
        *args,
        pvsd_views: tuple[str, ...] = ("full_solution",),
        pvsd_layer: int | None = None,
        pvsd_layer_fraction: str = "quarter",
        pvsd_alpha: float = 1.0,
        pvsd_purification: str = "contrast",
        pvsd_steer_scope: str = "completion",
        pvsd_extract_micro_batch: int = 8,
        pvsd_top_k_heads: int = 10,
        pvsd_pie_enabled: bool = True,
        pvsd_pie_every: int = 100,
        pvsd_pie_num_examples: int = 2,
        pvsd_pie_head_chunk: int = 8,
        pvsd_pie_layers: tuple[int, ...] | None = None,
        pvsd_pie_score: str = "inverse_kl",
        pvsd_log_advantage: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.multi_view_mode != "single":
            raise ValueError(
                "PVSDTrainer replaces AVSD's output-space aggregation, so it requires "
                "multi_view_mode='single'. Use --pvsd_views to set the PVSD view set."
            )
        if self.use_thinking_machines_loss:
            raise ValueError("PVSDTrainer supports the JSD/KL loss path only; disable use_tinker_loss.")
        if self.reason_first:
            raise ValueError("PVSDTrainer does not support reason_first.")
        if self.fixed_teacher or self.use_ema_teacher:
            raise ValueError(
                "PVSD computes the privilege vector and the teacher distribution from the "
                "current parameters theta, so fixed_teacher/use_ema_teacher are not applicable."
            )
        if not getattr(self.args, "disable_dropout", True):
            # The student pass and the steered pass are a paired comparison: they must
            # differ only by the injection. Active dropout would give them different
            # masks, so the loss would measure dropout noise instead of steering.
            raise ValueError(
                "PVSD requires disable_dropout=True: the student and the steered teacher "
                "must differ only by the injected vector."
            )

        views = tuple(pvsd_views)
        if not views:
            raise ValueError("pvsd_views must contain at least one view.")
        for view in views:
            if view not in VIEW_TYPES:
                raise ValueError(f"Unknown PVSD view '{view}'. Supported: {', '.join(VIEW_TYPES)}")
        if pvsd_steer_scope not in STEER_SCOPES:
            raise ValueError(f"pvsd_steer_scope must be one of: {', '.join(STEER_SCOPES)}")
        if pvsd_purification not in PURIFICATION_MODES:
            raise ValueError(f"pvsd_purification must be one of: {', '.join(PURIFICATION_MODES)}")
        if pvsd_pie_score not in PIE_SCORE_MODES:
            raise ValueError(f"pvsd_pie_score must be one of: {', '.join(PIE_SCORE_MODES)}")
        if pvsd_top_k_heads <= 0:
            raise ValueError("pvsd_top_k_heads must be positive.")
        if pvsd_pie_num_examples <= 0:
            raise ValueError("pvsd_pie_num_examples must be positive.")
        if pvsd_pie_head_chunk <= 0:
            raise ValueError("pvsd_pie_head_chunk must be positive.")

        collator_views = getattr(self.data_collator, "pvsd_views", None)
        if collator_views is None:
            # The batch is checked again at the first step, which is the real
            # safety net; a wrapped collator should not block startup here.
            print(
                "[PVSD] WARNING: could not read pvsd_views off the data collator "
                f"({type(self.data_collator).__name__}). PVSD needs a "
                "SelfDistillationDataCollator built with pvsd_views=..."
            )
        elif tuple(collator_views) != views:
            raise ValueError(
                f"pvsd_views {views} does not match the data collator's views {tuple(collator_views)}."
            )

        self.pvsd_views = views
        self.pvsd_alpha = float(pvsd_alpha)
        self.pvsd_purification = pvsd_purification
        self.pvsd_steer_scope = pvsd_steer_scope
        self.pvsd_extract_micro_batch = int(pvsd_extract_micro_batch)
        self.pvsd_top_k_heads = int(pvsd_top_k_heads)
        self.pvsd_pie_enabled = bool(pvsd_pie_enabled)
        self.pvsd_pie_every = int(pvsd_pie_every)
        self.pvsd_pie_num_examples = int(pvsd_pie_num_examples)
        self.pvsd_pie_head_chunk = int(pvsd_pie_head_chunk)
        self.pvsd_pie_score = pvsd_pie_score
        self.pvsd_log_advantage = bool(pvsd_log_advantage)

        self._pvsd_topology = describe_model(self.model)
        num_layers = self._pvsd_topology.num_layers

        if pvsd_layer is None:
            divisors = {"quarter": 4, "third": 3, "half": 2}
            if pvsd_layer_fraction not in divisors:
                raise ValueError(f"pvsd_layer_fraction must be one of: {', '.join(divisors)}")
            pvsd_layer = num_layers // divisors[pvsd_layer_fraction]
        self.pvsd_layer = int(pvsd_layer)
        if not 0 <= self.pvsd_layer < num_layers:
            raise ValueError(f"pvsd_layer must be in [0, {num_layers}); got {self.pvsd_layer}.")

        if pvsd_pie_layers is None:
            self.pvsd_pie_layers = tuple(range(num_layers))
        else:
            self.pvsd_pie_layers = tuple(sorted({int(layer) for layer in pvsd_pie_layers}))
            if not self.pvsd_pie_layers:
                raise ValueError("pvsd_pie_layers must name at least one layer.")
            for layer_idx in self.pvsd_pie_layers:
                if not 0 <= layer_idx < num_layers:
                    raise ValueError(f"pvsd_pie_layers contains out-of-range layer {layer_idx}.")

        num_candidate_heads = len(self.pvsd_pie_layers) * self._pvsd_topology.num_heads
        if self.pvsd_pie_enabled and self.pvsd_top_k_heads > num_candidate_heads:
            raise ValueError(
                f"pvsd_top_k_heads ({self.pvsd_top_k_heads}) exceeds the number of candidate "
                f"heads ({num_candidate_heads})."
            )

        # One head set A^(m) per view, as PIE is defined per view.
        self._pvsd_head_sets: dict[str, tuple[tuple[int, int], ...]] | None = None
        self._pvsd_head_sets_step: int | None = None
        if not self.pvsd_pie_enabled:
            # No-PIE ablation: read from every head of the injection layer.
            shared = all_heads_in_layers(self._pvsd_topology, [self.pvsd_layer])
            self._pvsd_head_sets = {view: shared for view in views}
            self._pvsd_head_sets_step = -1

        if abs(self.beta - 1.0) > 1e-6:
            print(
                f"[PVSD] WARNING: beta={self.beta}. The method is defined with reverse KL "
                "D_KL(student || steered teacher), which is beta=1.0 in this loss. "
                "Pass --beta 1.0 unless you are deliberately running a JSD ablation."
            )
        if self.jsd_token_clip is not None:
            print(
                f"[PVSD] WARNING: jsd_token_clip={self.jsd_token_clip} is active. It zeroes the "
                "gradient exactly where the steered teacher disagrees most with the student. "
                "Pass --jsd_token_clip 0 unless you are deliberately ablating it."
            )
        if self.pvsd_purification != "contrast":
            print(
                f"[PVSD] Running the '{self.pvsd_purification}' purification ablation, not the "
                "method itself."
            )

        print(f"\n{'=' * 80}")
        print("PVSD MODE ENABLED (privilege-vector self-distillation)")
        print(f"Views: {self.pvsd_views}")
        print(f"Injection layer: {self.pvsd_layer} / {num_layers}")
        print(f"Alpha: {self.pvsd_alpha}")
        print(f"Purification: {self.pvsd_purification}")
        print(f"Steer scope: {self.pvsd_steer_scope}")
        print(f"Model topology: {self._pvsd_topology}")
        if self.pvsd_pie_enabled:
            print(f"PIE: top-{self.pvsd_top_k_heads} heads per view, every {self.pvsd_pie_every} steps")
            print(f"PIE candidate layers: {len(self.pvsd_pie_layers)} -> {num_candidate_heads} heads")
            print(
                "PIE cost per refresh: ~"
                f"{len(views) * estimate_pie_forward_passes(self.pvsd_pie_num_examples, num_candidate_heads, self.pvsd_pie_head_chunk)}"
                " prompt-only forward passes"
            )
        else:
            print(f"PIE disabled (ablation): reading all heads of layer {self.pvsd_layer}")
        print(f"{'=' * 80}\n")

    # ------------------------------------------------------------------
    # Stage 1: head localisation
    # ------------------------------------------------------------------

    def _zero_stage_3(self) -> bool:
        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        return deepspeed_plugin is not None and getattr(deepspeed_plugin, "zero_stage", None) == 3

    def _should_refresh_head_sets(self) -> bool:
        step = int(self.state.global_step)
        if self._pvsd_head_sets is None:
            return True
        if not self.pvsd_pie_enabled or self.pvsd_pie_every <= 0:
            return False
        if self._pvsd_head_sets_step == step:
            # Gradient accumulation calls compute_loss several times per step.
            return False
        return step % self.pvsd_pie_every == 0

    def _refresh_head_sets(self, model, inputs) -> None:
        step = int(self.state.global_step)
        if not self.pvsd_pie_enabled:
            self._pvsd_head_sets_step = step
            return

        # ZeRO-3 shards parameters, so a forward on rank 0 alone would deadlock on
        # the parameter all-gather. There, every rank runs PIE on its own batch and
        # rank 0's head sets are broadcast; otherwise only rank 0 pays the cost.
        run_locally = self._zero_stage_3() or self.accelerator.is_main_process
        results = {}
        if run_locally:
            was_training = model.training
            model.eval()
            try:
                for view in self.pvsd_views:
                    results[view] = compute_privilege_indirect_effect(
                        model,
                        self._pvsd_topology,
                        inputs[f"pvsd_{view}_input_ids"],
                        inputs[f"pvsd_{view}_attention_mask"],
                        inputs[f"pvsd_{view}_corrupt_input_ids"][:, 0, :],
                        inputs[f"pvsd_{view}_corrupt_attention_mask"][:, 0, :],
                        top_k_heads=self.pvsd_top_k_heads,
                        candidate_layers=self.pvsd_pie_layers,
                        head_chunk_size=self.pvsd_pie_head_chunk,
                        max_examples=self.pvsd_pie_num_examples,
                        score_mode=self.pvsd_pie_score,
                    )
            finally:
                model.train(was_training)
                empty_cache()

        payload = [{view: result.top_heads for view, result in results.items()} if results else None]
        if self.accelerator.num_processes > 1:
            payload = broadcast_object_list(payload, from_process=0)
        head_sets = payload[0]
        if not head_sets:
            raise RuntimeError("PIE calibration produced no head sets on the main process.")

        self._pvsd_head_sets = {
            view: normalize_head_set(head_sets[view], self._pvsd_topology)
            for view in self.pvsd_views
        }
        self._pvsd_head_sets_step = step

        for view, result in results.items():
            for key, value in result.as_log_dict(prefix=f"pvsd/pie/{view}").items():
                self._append_metric("train", key, value)
        self._log_head_set_overlap()
        if self.accelerator.is_main_process:
            for view, heads in self._pvsd_head_sets.items():
                print(f"[PVSD] PIE step {step} view={view}: top-{len(heads)} heads {list(heads)}")

    def _log_head_set_overlap(self) -> None:
        """Cross-view head overlap: the mechanistic motivation for fusing views."""

        views = self.pvsd_views
        for first in range(len(views)):
            for second in range(first + 1, len(views)):
                left, right = views[first], views[second]
                self._append_metric(
                    "train",
                    f"pvsd/head_jaccard/{left}__{right}",
                    head_set_jaccard(self._pvsd_head_sets[left], self._pvsd_head_sets[right]),
                )

    # ------------------------------------------------------------------
    # Stage 2: privilege vector
    # ------------------------------------------------------------------

    def _require_pvsd_batch(self, inputs) -> tuple[str, ...]:
        if "pvsd_view_names" not in inputs:
            raise ValueError(
                "the batch has no PVSD prompt sets. Build the trainer with a "
                "SelfDistillationDataCollator(..., pvsd_views=..., pvsd_num_corrupt=...) "
                "so each batch carries the privileged and corrupted contexts."
            )
        view_names = tuple(inputs["pvsd_view_names"])
        if view_names != self.pvsd_views:
            raise ValueError(
                f"batch views {view_names} do not match trainer views {self.pvsd_views}."
            )
        return view_names

    def _build_privilege_vector(self, model, inputs) -> torch.Tensor:
        view_names = self._require_pvsd_batch(inputs)

        fused, per_view = build_privilege_vectors_from_batch(
            model,
            self._pvsd_topology,
            view_names,
            inputs,
            self._pvsd_head_sets,
            micro_batch_size=self.pvsd_extract_micro_batch,
            purification=self.pvsd_purification,
        )
        for view_vectors in per_view:
            self._log_vector_diagnostics(inputs, view_vectors)
        empty_cache()

        self._append_metric("train", "pvsd/fused_norm", fused.norm(dim=-1))
        return fused

    def _log_vector_diagnostics(self, inputs, view_vectors):
        view = view_vectors.view
        prefix = f"pvsd/{view}"
        raw_vector = view_vectors.raw
        transfer_vector = view_vectors.transfer
        corrupt_mean = view_vectors.corrupt_mean
        raw_norm = raw_vector.norm(dim=-1)

        self._append_metric("train", f"{prefix}/raw_norm", raw_norm)
        self._append_metric("train", f"{prefix}/corrupt_norm", corrupt_mean.norm(dim=-1))
        self._append_metric("train", f"{prefix}/transfer_norm", transfer_vector.norm(dim=-1))
        self._append_metric("train", f"{prefix}/num_heads", float(len(view_vectors.heads)))
        self._append_metric(
            "train",
            f"{prefix}/heads_mean_layer",
            float(sum(layer for layer, _ in view_vectors.heads) / max(1, len(view_vectors.heads))),
        )
        # cos(raw, corrupt) near 1 means the raw vector is almost entirely the view
        # template: the purified vector is then a small residual and the method has
        # little content to transfer. This is the key sanity metric.
        self._append_metric(
            "train",
            f"{prefix}/cos_raw_corrupt",
            cosine_similarity_rowwise(raw_vector, corrupt_mean),
        )
        self._append_metric(
            "train",
            f"{prefix}/transfer_ratio",
            transfer_vector.norm(dim=-1) / raw_norm.clamp_min(1e-8),
        )
        real_lengths = inputs.get(f"pvsd_{view}_lengths")
        corrupt_lengths = inputs.get(f"pvsd_{view}_corrupt_lengths")
        if real_lengths is not None and corrupt_lengths is not None:
            delta = (corrupt_lengths.float() - real_lengths.float().unsqueeze(-1)).abs()
            self._append_metric("train", f"{prefix}/corrupt_len_delta", delta)

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------

    def _compute_loss_single_view(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        self._require_pvsd_batch(inputs)

        student_prompt_len = int(inputs["student_prompt_length"])
        student_ids = inputs["student_input_ids"]
        student_mask = inputs["student_attention_mask"]
        shifted_labels = inputs["labels"][:, student_prompt_len:]
        sampled_token_ids = student_ids[:, student_prompt_len:]
        # Prompts are right-padded and the rollout is concatenated after that
        # padding, so explicit position ids are required for the training forward
        # pass to see the same positions the rollout was sampled at.
        position_ids = position_ids_from_mask(student_mask)

        outputs_student = model(
            input_ids=student_ids,
            attention_mask=student_mask,
            position_ids=position_ids,
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
        del outputs_student
        empty_cache()

        if self._should_refresh_head_sets():
            self._refresh_head_sets(model, inputs)

        with torch.no_grad():
            privilege_vector = self._build_privilege_vector(model, inputs)

        steer_start = 0 if self.pvsd_steer_scope == "all" else student_prompt_len - 1
        with torch.no_grad(), inject_at_layer(
            model,
            self.pvsd_layer,
            privilege_vector,
            alpha=self.pvsd_alpha,
            start_index=steer_start,
        ) as hook:
            outputs_steered = model(
                input_ids=student_ids,
                attention_mask=student_mask,
                position_ids=position_ids,
            )
            steered_logits = outputs_steered.logits[:, student_prompt_len - 1 : -1, :]
            del outputs_steered
        if hook.call_count == 0:
            raise RuntimeError(
                f"PVSD injection hook never fired on layer {self.pvsd_layer}; "
                "the steered pass would be identical to the student pass."
            )

        if self.pvsd_log_advantage:
            self._log_steering_advantage(
                student_logits, steered_logits, sampled_token_ids, shifted_labels
            )

        loss = self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=steered_logits,
            labels=shifted_labels,
            beta=self.beta,
            temperature=self.temperature,
            top_k=self.top_k_loss,
            token_clip=self.jsd_token_clip,
        )

        del student_logits, steered_logits, privilege_vector
        empty_cache()

        if return_outputs:
            class MinimalOutput:
                def __init__(self, value):
                    self.loss = value

            return loss, MinimalOutput(loss)
        return loss

    @torch.no_grad()
    def _log_steering_advantage(self, student_logits, steered_logits, sampled_token_ids, shifted_labels):
        """Does the steered teacher actually prefer the rollout more than the student?"""

        mask = shifted_labels != -100
        if not torch.any(mask):
            return
        student_log_probs = sampled_token_log_probs(
            student_logits.detach(), sampled_token_ids, self.temperature
        )
        steered_log_probs = sampled_token_log_probs(
            steered_logits, sampled_token_ids, self.temperature
        )
        advantage = steered_log_probs - student_log_probs
        self._append_metric("train", "pvsd/steer_advantage", advantage[mask])
        self._append_metric(
            "train",
            "pvsd/steer_advantage_positive_frac",
            (advantage[mask] > 0).float(),
        )
        self._append_metric("train", "pvsd/student_logprob", student_log_probs[mask])
        self._append_metric("train", "pvsd/steered_logprob", steered_log_probs[mask])
        del student_log_probs, steered_log_probs, advantage
