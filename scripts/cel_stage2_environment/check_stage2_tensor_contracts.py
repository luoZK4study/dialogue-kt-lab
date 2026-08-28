#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dialogue_kt.cel_methods import (
    apply_rationale_injection,
    apply_stage2_environment_injection,
    build_stage2_dual_path_representations,
    build_environment_generator,
    build_stage2_split_mask,
    register_cel_dual_path_hook,
    register_cel_injection_hook,
)
from dialogue_kt.training import (
    CELProbabilityCalibrator,
    _cel_can_skip_backbone_grad_path,
    _clear_pending_cel_hooks,
    _get_lmkt_loss_cel_dual_path,
    _get_lmkt_loss_cel_dual_path_serial,
    _stage2_serial_path_required,
)


class TinyCausalBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mix = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        cumulative = hidden_states.cumsum(dim=1)
        counts = torch.arange(1, hidden_states.shape[1] + 1, device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states + self.mix(cumulative / counts.view(1, -1, 1))


class TinyCausalModel(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([TinyCausalBlock(hidden_dim)])

    def forward(self, hidden_states: torch.Tensor):
        return self.layers[0](hidden_states)


class TinyCheckpointCausalModel(TinyCausalModel):
    def forward(self, hidden_states: torch.Tensor):
        return checkpoint(self.layers[0], hidden_states, use_reentrant=True)


class TinySelector(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        hidden_states,
        answer_state=None,
        kc_state=None,
        context_mask=None,
    ):
        signal = torch.tanh(self.proj(hidden_states).squeeze(-1))
        return signal * context_mask.to(dtype=signal.dtype)


class TinyDualPathLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([TinyCausalBlock(hidden_dim)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    @property
    def dtype(self):
        return self.embedding.weight.dtype

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        hidden_states = self.embedding(input_ids)
        hidden_states = self.layers[0](hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


def check_serial_gradient_equivalence(device: torch.device) -> float:
    torch.manual_seed(1221)
    hidden_dim = 8
    model = TinyDualPathLM(vocab_size=13, hidden_dim=hidden_dim)
    model._cel_selector = TinySelector(hidden_dim)
    model._cel_environment = build_environment_generator(
        "mlp",
        hidden_dim,
        env_hidden_dim=12,
        dropout=0.0,
    )
    nn.init.normal_(model._cel_environment.up.weight, mean=0.0, std=0.01)
    model._cel_calibrator = CELProbabilityCalibrator("bias", init_bias=0.1)
    model._cel_metrics = {}
    model._cel_last_outputs = {}
    model._cel_pending_hook_handles = []
    model._cel_active_train_mode = "joint"
    model._cel_stage2_training_progress = 1.0
    model = model.to(device)
    serial_model = copy.deepcopy(model)

    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.ones(2, 8, dtype=torch.long, device=device),
        "dialogue_token_ranges": [(1, 6), (1, 6)],
        "last_idxs": torch.tensor([6, 6], dtype=torch.long, device=device),
        "num_kcs": [1, 1],
        "labels": torch.tensor([1.0, 0.0], device=device),
    }
    args = SimpleNamespace(
        cel_mode="mlp",
        cel_layer_idx=-1,
        cel_env_beta=0.1,
        cel_env_output_postprocess="centered_rms",
        cel_env_output_ratio=1.0,
        cel_stage2_phase="joint",
        cel_stage2_lambda_r=1.0,
        cel_stage2_lambda_m=1.0,
        cel_stage2_lambda_cons=0.2,
        cel_stage2_beta_start_ratio=0.2,
        cel_stage2_consistency_ramp_fraction=0.25,
        agg="mean-ar",
    )

    model._cel_serial_joint_backward = False
    model._cel_loss_backward_done = False
    expanded_loss, _, _ = _get_lmkt_loss_cel_dual_path(
        model,
        batch,
        true_token=1,
        false_token=2,
        args=args,
        selector=model._cel_selector,
    )
    expanded_loss.backward()
    _clear_pending_cel_hooks(model)

    serial_model._cel_serial_joint_backward = True
    serial_model._cel_serial_backward_scale = 1.0
    serial_model._cel_loss_backward_done = False
    serial_loss, _, _ = _get_lmkt_loss_cel_dual_path_serial(
        serial_model,
        batch,
        true_token=1,
        false_token=2,
        args=args,
        selector=serial_model._cel_selector,
    )

    assert torch.allclose(expanded_loss.detach(), serial_loss, atol=1e-6, rtol=1e-5)
    expanded_parameters = dict(model.named_parameters())
    serial_parameters = dict(serial_model.named_parameters())
    assert expanded_parameters.keys() == serial_parameters.keys()
    maximum_difference = 0.0
    for name, expanded_parameter in expanded_parameters.items():
        serial_parameter = serial_parameters[name]
        expanded_gradient = expanded_parameter.grad
        serial_gradient = serial_parameter.grad
        if expanded_gradient is None or serial_gradient is None:
            assert expanded_gradient is None and serial_gradient is None
            continue
        difference = (
            expanded_gradient.detach().float()
            - serial_gradient.detach().float()
        ).abs().max().item()
        maximum_difference = max(maximum_difference, difference)
        assert torch.allclose(
            expanded_gradient.detach().float(),
            serial_gradient.detach().float(),
            atol=1e-5,
            rtol=1e-4,
        ), f"serial gradient mismatch for {name}: {difference}"

    low_risk_batch = {"input_ids": torch.empty(2, 8, dtype=torch.long)}
    high_risk_batch = {"input_ids": torch.empty(2, 2000, dtype=torch.long)}
    assert not _stage2_serial_path_required(serial_model, low_risk_batch)
    assert _stage2_serial_path_required(serial_model, high_risk_batch)
    return maximum_difference


def check_device(device: torch.device) -> None:
    torch.manual_seed(221)
    # B warmup still has a differentiable B-to-suffix path, but it must not
    # preserve gradients through the frozen Qwen prefix.
    assert _cel_can_skip_backbone_grad_path(
        SimpleNamespace(
            cel_train_calibrator_only=False,
            cel_train_environment_only=True,
        )
    )
    assert _cel_can_skip_backbone_grad_path(
        SimpleNamespace(
            cel_train_calibrator_only=True,
            cel_train_environment_only=False,
        )
    )
    assert not _cel_can_skip_backbone_grad_path(
        SimpleNamespace(
            cel_train_calibrator_only=False,
            cel_train_environment_only=False,
        )
    )
    batch_size, seq_len, hidden_dim = 2, 12, 16
    hidden = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
    signal = torch.linspace(-0.8, 0.9, steps=batch_size * seq_len, device=device).reshape(batch_size, seq_len)
    context_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    context_mask[0, 1:11] = True
    context_mask[1, 2:10] = True

    split_mask = build_stage2_split_mask(signal, context_mask, mode="topk_abs", topk_ratio=0.2)
    assert split_mask[0].sum().item() == 2
    assert split_mask[1].sum().item() == 2
    assert torch.all(split_mask[~context_mask] == 0)

    stage1 = apply_rationale_injection(hidden, signal, gamma=0.3, context_mask=context_mask)

    # Target dual-path contract: continuous complementary decomposition and a
    # capacity-adequate sequence B, independent of the historical top-k path.
    target_environment = build_environment_generator(
        "contextual_transformer",
        hidden_dim,
        env_hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        output_init_std=0.01,
    ).to(device=device, dtype=torch.float32)
    target_representations = build_stage2_dual_path_representations(
        hidden,
        signal,
        target_environment,
        context_mask,
        beta=0.1,
        environment_dtype=torch.float32,
        environment_output_postprocess="centered_rms",
        environment_output_ratio=1.0,
    )
    assert target_representations["h_r"].shape == hidden.shape
    assert target_representations["h_n"].shape == hidden.shape
    assert target_representations["h_nb"].shape == hidden.shape
    assert target_representations["h_m"].shape == hidden.shape
    assert torch.allclose(
        target_representations["h_r"] + target_representations["h_n"],
        hidden,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.all(target_representations["h_r"][~context_mask] == hidden[~context_mask])
    assert torch.all(target_representations["h_n"][~context_mask] == 0)
    assert torch.all(target_representations["h_nb"][~context_mask] == 0)
    assert torch.isfinite(target_representations["h_m"]).all()
    target_stats = target_representations["stats"]
    assert target_stats["complement_relative_rms_error"].item() < 1e-6
    assert target_stats["weight_complement_max_abs_error"].item() < 1e-6

    # The raw maximum residual scales with large hidden-state outliers in BF16.
    # The relative RMS and weight errors remain the audit-stable contract checks.
    with torch.no_grad():
        low_precision_representations = build_stage2_dual_path_representations(
            hidden.detach().to(dtype=torch.bfloat16),
            signal.detach().to(dtype=torch.bfloat16),
            target_environment,
            context_mask,
            beta=0.1,
            environment_dtype=torch.float32,
            environment_output_postprocess="centered_rms",
            environment_output_ratio=1.0,
        )
    low_precision_stats = low_precision_representations["stats"]
    assert torch.isfinite(low_precision_stats["complement_max_abs_error"])
    assert low_precision_stats["complement_relative_rms_error"].item() <= 0.01
    assert low_precision_stats["weight_complement_max_abs_error"].item() <= 0.01
    target_loss = target_representations["h_m"].float().square().mean()
    target_loss.backward(retain_graph=True)
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in target_environment.parameters()
    )
    for mode in ("shuffle", "mlp", "transformer"):
        generator = build_environment_generator(
            mode,
            hidden_dim,
            env_hidden_dim=16,
            num_layers=1,
            num_heads=4,
            ffn_dim=32,
            dropout=0.0,
        ).to(device)
        injected_zero, details_zero = apply_stage2_environment_injection(
            hidden,
            signal,
            generator,
            context_mask,
            gamma=0.3,
            beta=0.0,
            split_mode="topk_abs",
            topk_ratio=0.2,
            sigmoid_temperature=5.0,
        )
        assert torch.allclose(injected_zero, stage1, atol=1e-5, rtol=1e-5)
        assert torch.isfinite(details_zero["environment"]).all()

        injected, details = apply_stage2_environment_injection(
            hidden,
            signal,
            generator,
            context_mask,
            gamma=0.3,
            beta=0.1,
            split_mode="topk_abs",
            topk_ratio=0.2,
            sigmoid_temperature=5.0,
        )
        assert injected.shape == hidden.shape
        assert torch.isfinite(injected).all()
        assert torch.all(details["environment"][~context_mask] == 0)
        loss = injected.float().square().mean()
        loss.backward(retain_graph=True)
        if mode != "shuffle":
            named_grads = {
                name: param.grad
                for name, param in generator.named_parameters()
                if param.requires_grad
            }
            assert named_grads
            assert all(grad is not None and torch.isfinite(grad).all() for grad in named_grads.values())
            if mode == "mlp":
                assert named_grads["up.weight"].abs().sum().item() > 0
            else:
                assert named_grads["down.weight"].abs().sum().item() > 0
                assert any(
                    grad.abs().sum().item() > 0
                    for name, grad in named_grads.items()
                    if name.startswith("encoder.")
                )
        if mode == "transformer":
            normalized, normalized_details = apply_stage2_environment_injection(
                hidden,
                signal,
                generator,
                context_mask,
                gamma=0.3,
                beta=0.1,
                split_mode="topk_abs",
                topk_ratio=0.2,
                sigmoid_temperature=5.0,
                environment_output_postprocess="centered_rms",
                environment_output_ratio=0.1,
            )
            normalized_environment = normalized_details["environment"].float()
            normalized_valid = normalized_details["non_rationale_mask"].bool()
            normalized_input = hidden.float() * normalized_valid.unsqueeze(-1)
            environment_rms = normalized_environment[normalized_valid].square().mean().sqrt()
            input_rms = normalized_input[normalized_valid].square().mean().sqrt()
            assert torch.allclose(environment_rms / input_rms, torch.tensor(0.1, device=device), atol=1e-4)
            pooled_environment = (
                normalized_environment * normalized_valid.to(dtype=torch.float32).unsqueeze(-1)
            ).sum(dim=1)
            assert pooled_environment.abs().max().item() < 1e-4
            normalized.float().square().mean().backward(retain_graph=True)

    shuffle = build_environment_generator("shuffle", hidden_dim, shuffle_seed=221).to(device)
    non_rationale_mask = context_mask.float() * (1.0 - split_mask)
    shuffled = shuffle(hidden.detach() * non_rationale_mask.unsqueeze(-1), non_rationale_mask)
    valid_values = hidden.detach()[non_rationale_mask.bool()]
    shuffled_values = shuffled[non_rationale_mask.bool()]
    assert torch.allclose(
        valid_values.float().sort(dim=0).values,
        shuffled_values.float().sort(dim=0).values,
        atol=1e-5,
        rtol=1e-5,
    )
    if valid_values.shape[0] > 1:
        assert not torch.equal(valid_values, shuffled_values)

    causal_hidden = torch.randn(1, 8, hidden_dim, device=device)
    causal_model = TinyCausalModel(hidden_dim).to(device)
    causal_selector = TinySelector(hidden_dim).to(device)
    tiny_batch = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long, device=device),
        "dialogue_token_ranges": [(1, 6)],
        "last_idxs": torch.tensor([6], device=device),
    }

    baseline_prediction = causal_model(causal_hidden)[:, 6].detach()
    post_model = TinyCausalModel(hidden_dim).to(device)
    post_model.load_state_dict(causal_model.state_dict())
    post_selector = TinySelector(hidden_dim).to(device)
    post_selector.load_state_dict(causal_selector.state_dict())
    post_hidden = causal_hidden.detach().clone().requires_grad_(True)
    handle, _ = register_cel_injection_hook(
        post_model,
        tiny_batch,
        post_selector,
        mode="task_conditioned",
        layer_idx=-1,
        hook_site="last_block",
        hook_timing="post_block",
        gamma=0.3,
        use_norm=False,
        application_mode="token_residual",
    )
    try:
        post_prediction = post_model(post_hidden)[:, 6]
        post_prediction.square().mean().backward()
    finally:
        handle.remove()
    assert torch.equal(post_prediction.detach(), baseline_prediction)
    post_gradient = post_selector.proj.weight.grad
    assert post_gradient is None or post_gradient.abs().sum().item() == 0

    pre_model = TinyCausalModel(hidden_dim).to(device)
    pre_model.load_state_dict(causal_model.state_dict())
    pre_selector = TinySelector(hidden_dim).to(device)
    pre_selector.load_state_dict(causal_selector.state_dict())
    pre_hidden = causal_hidden.detach().clone().requires_grad_(True)
    handle, _ = register_cel_injection_hook(
        pre_model,
        tiny_batch,
        pre_selector,
        mode="task_conditioned",
        layer_idx=-1,
        hook_site="last_block",
        hook_timing="pre_block",
        gamma=0.3,
        use_norm=False,
        application_mode="token_residual",
    )
    try:
        pre_prediction = pre_model(pre_hidden)[:, 6]
        pre_prediction.square().mean().backward()
    finally:
        handle.remove()
    assert not torch.allclose(pre_prediction.detach(), baseline_prediction)
    assert pre_selector.proj.weight.grad is not None
    assert pre_selector.proj.weight.grad.abs().sum().item() > 0

    # The dual hook expands the final-block batch while duplicating attention
    # kwargs; all branches must use the same downstream module parameters.
    dual_model = TinyCausalModel(hidden_dim).to(device)
    dual_selector = TinySelector(hidden_dim).to(device)
    dual_environment = build_environment_generator(
        "contextual_transformer",
        hidden_dim,
        env_hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        output_init_std=0.01,
    ).to(device=device, dtype=torch.float32)
    dual_batch = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long, device=device),
        "dialogue_token_ranges": [(1, 6)],
        "last_idxs": torch.tensor([6], device=device),
        "num_kcs": [1],
    }
    dual_hidden = torch.randn(1, 8, hidden_dim, device=device, requires_grad=True)
    dual_handle, dual_captured = register_cel_dual_path_hook(
        dual_model,
        dual_batch,
        dual_selector,
        dual_environment,
        mode="task_conditioned",
        layer_idx=-1,
        beta=0.1,
        include_reversal=True,
        environment_output_postprocess="centered_rms",
        environment_output_ratio=1.0,
    )
    try:
        dual_output = dual_model(dual_hidden)
        assert dual_output.shape[0] == 3
        assert dual_captured["branch_names"] == ("evidence", "mixed", "non_evidence")
        dual_output.square().mean().backward()
    finally:
        dual_handle.remove()
    assert dual_selector.proj.weight.grad is not None
    assert dual_selector.proj.weight.grad.abs().sum().item() > 0
    assert any(
        param.grad is not None and param.grad.abs().sum().item() > 0
        for param in dual_environment.parameters()
    )

    # B-only warmup executes the evidence branch separately under no_grad, so
    # an evidence-only hook must not invoke B or expand the suffix batch.
    evidence_only_model = TinyCausalModel(hidden_dim).to(device)
    evidence_only_selector = TinySelector(hidden_dim).to(device)
    evidence_only_environment = build_environment_generator(
        "contextual_transformer",
        hidden_dim,
        env_hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        output_init_std=0.01,
    ).to(device=device, dtype=torch.float32)
    environment_calls = []
    environment_counter = evidence_only_environment.register_forward_hook(
        lambda *_: environment_calls.append(True)
    )
    evidence_handle, evidence_captured = register_cel_dual_path_hook(
        evidence_only_model,
        dual_batch,
        evidence_only_selector,
        evidence_only_environment,
        mode="task_conditioned",
        layer_idx=-1,
        beta=0.1,
        branch_names=("evidence",),
        environment_output_postprocess="centered_rms",
        environment_output_ratio=1.0,
    )
    try:
        evidence_output = evidence_only_model(
            torch.randn(1, 8, hidden_dim, device=device)
        )
        assert evidence_output.shape[0] == 1
        assert evidence_captured["branch_names"] == ("evidence",)
        assert evidence_captured["stage2_stats"] == {}
    finally:
        evidence_handle.remove()
        environment_counter.remove()
    assert not environment_calls

    dual_checkpoint_model = TinyCheckpointCausalModel(hidden_dim).to(device)
    dual_checkpoint_selector = TinySelector(hidden_dim).to(device)
    dual_checkpoint_environment = build_environment_generator(
        "contextual_transformer",
        hidden_dim,
        env_hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        output_init_std=0.01,
    ).to(device=device, dtype=torch.float32)
    dual_checkpoint_hidden = torch.randn(1, 8, hidden_dim, device=device, requires_grad=True)
    checkpoint_handle, _ = register_cel_dual_path_hook(
        dual_checkpoint_model,
        dual_batch,
        dual_checkpoint_selector,
        dual_checkpoint_environment,
        mode="task_conditioned",
        layer_idx=-1,
        beta=0.1,
        include_reversal=False,
        environment_output_postprocess="centered_rms",
        environment_output_ratio=1.0,
    )
    try:
        checkpoint_output = dual_checkpoint_model(dual_checkpoint_hidden)
        assert checkpoint_output.shape[0] == 2
        checkpoint_output.square().mean().backward()
    finally:
        checkpoint_handle.remove()
    assert dual_checkpoint_selector.proj.weight.grad is not None
    assert dual_checkpoint_selector.proj.weight.grad.abs().sum().item() > 0
    assert any(
        param.grad is not None and param.grad.abs().sum().item() > 0
        for param in dual_checkpoint_environment.parameters()
    )

    checkpoint_hidden = torch.randn(1, 8, hidden_dim, device=device, requires_grad=True)
    checkpoint_model = TinyCheckpointCausalModel(hidden_dim).to(device)
    checkpoint_selector = TinySelector(hidden_dim).to(device, dtype=torch.float32)
    checkpoint_environment = build_environment_generator(
        "mlp", hidden_dim, env_hidden_dim=16, dropout=0.0
    ).to(device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        list(checkpoint_selector.parameters()) + list(checkpoint_environment.parameters()),
        lr=1e-3,
    )
    selector_before = checkpoint_selector.proj.weight.detach().clone()
    environment_before = checkpoint_environment.up.weight.detach().clone()
    handle, _ = register_cel_injection_hook(
        checkpoint_model,
        tiny_batch,
        checkpoint_selector,
        mode="task_conditioned",
        layer_idx=-1,
        hook_site="last_block",
        hook_timing="pre_block",
        gamma=0.3,
        use_norm=False,
        application_mode="token_residual",
        environment_generator=checkpoint_environment,
        environment_beta=0.1,
        environment_topk_ratio=0.2,
    )
    try:
        prediction = checkpoint_model(checkpoint_hidden)[:, 6].square().mean()
        prediction.backward()
    finally:
        # Reentrant checkpointing executes the hooked block again during
        # backward, so cleanup must happen after backward has completed.
        handle.remove()
    assert checkpoint_selector.proj.weight.grad is not None
    assert checkpoint_selector.proj.weight.grad.abs().sum().item() > 0
    assert checkpoint_environment.up.weight.grad is not None
    assert checkpoint_environment.up.weight.grad.abs().sum().item() > 0
    optimizer.step()
    assert not torch.equal(selector_before, checkpoint_selector.proj.weight.detach())
    assert not torch.equal(environment_before, checkpoint_environment.up.weight.detach())
    assert checkpoint_selector.proj.weight.dtype == torch.float32
    assert checkpoint_environment.up.weight.dtype == torch.float32

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        mixed_hidden = torch.randn(
            1,
            8,
            hidden_dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        mixed_signal = torch.linspace(-0.5, 0.5, steps=8, device=device).unsqueeze(0)
        mixed_mask = torch.zeros(1, 8, dtype=torch.bool, device=device)
        mixed_mask[:, 1:7] = True
        mixed_environment = build_environment_generator(
            "transformer",
            hidden_dim,
            env_hidden_dim=16,
            num_layers=1,
            num_heads=4,
            ffn_dim=32,
            dropout=0.0,
            output_init_std=0.01,
        ).to(device=device, dtype=torch.float32)
        mixed_injected, mixed_details = apply_stage2_environment_injection(
            mixed_hidden,
            mixed_signal,
            mixed_environment,
            mixed_mask,
            gamma=0.3,
            beta=0.1,
            split_mode="topk_abs",
            topk_ratio=0.2,
            sigmoid_temperature=5.0,
            environment_output_postprocess="centered_rms",
            environment_output_ratio=0.1,
        )
        assert mixed_injected.dtype == torch.bfloat16
        assert mixed_details["environment"].dtype == torch.float32
        assert torch.isfinite(mixed_injected).all()
        mixed_injected.float().square().mean().backward()
        assert mixed_environment.up.weight.grad is not None
        assert mixed_environment.up.weight.grad.abs().sum().item() > 0

    serial_gradient_difference = check_serial_gradient_equivalence(device)
    print(
        "Stage 2 serial/expanded gradient equivalence passed on "
        f"{device}: max_abs_diff={serial_gradient_difference:.8g}"
    )
    print(f"Stage 2 tensor contracts passed on {device}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    check_device(torch.device("cpu"))
    if args.cuda:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable")
        check_device(torch.device("cuda"))


if __name__ == "__main__":
    main()
