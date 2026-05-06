#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
LoRA-Structured Export for Swift Backward (Qwen2).

This module exports forward functions with intermediate outputs needed for
Swift-side structured backward computation. The key insight is:
- Export forward function that outputs layer input x (for LoRA backward)
- Swift computes structured backward, recomputing h = x @ A
- This achieves memory savings by not storing h in the exported graph

Note: Qwen2 differs from Qwen3 in that:
- No Q/K normalization layers
- Different attention structure
"""
from __future__ import annotations

import os
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.models.rope_utils import initialize_rope

from .modules import LazyLoadingMixin, LazyLoadingQuantizedLinear, LazyLoadingRMSNorm as RMSNorm
from .qwen2 import QLoRAModelArgs
from .qwen2_structured import (
    StructuredTransformerBlock,
    StructuredAttention,
    StructuredMLP,
)
from .runner import RunFunctionConfig
from .utils import save_params, get_grad_name


def export_transformer_block_for_swift_backward(
    args: QLoRAModelArgs,
    module: nn.Module,
    variable_names: dict[str, str],
    output_dir: str,
    save_params_name: str,
    batch_size: int = 1,
    context_length: int = 256,
    shapeless: bool = False,
    module_name: str = "transformer_block",
    param_name_formatter: Callable[[str], str] = lambda x: x,
) -> RunFunctionConfig:
    """
    Export transformer block for Swift-side structured backward (Qwen2).

    The forward function outputs:
    - y: Layer output (for next layer / loss)
    - x_saved: Layer input (for Swift to compute structured backward)
    - attn_weights: Softmax output (for attention backward)
    - gate_input: MLP input (for SiLU backward)

    Swift will use these to compute structured backward without needing
    the exported backward graph.
    """
    assert "x" in variable_names and "y" in variable_names

    # Get all parameters
    trainable_param_names, trainable_params = zip(*tree_flatten(module.trainable_parameters()))
    all_param_names, all_params = zip(*tree_flatten(module.parameters()))

    frozen_param_names, frozen_params = [], []
    for name, param in zip(all_param_names, all_params):
        if name not in trainable_param_names:
            frozen_param_names.append(name)
            frozen_params.append(param)

    flatten_param_names = trainable_param_names + tuple(frozen_param_names)
    flatten_params = trainable_params + tuple(frozen_params)
    dtype = trainable_params[0].dtype

    # Create structured module (has LoRA built-in)
    structured_module = StructuredTransformerBlock(args)

    def forward_with_intermediates(*inputs):
        """
        Forward pass that returns intermediates needed for Swift backward.

        Key insight: We save inputs to each component, not LoRA intermediates.
        Swift will recompute h = x @ A during backward (LoRA Structured).
        """
        x, *flat_params = inputs
        params = tree_unflatten(list(zip(flatten_param_names, flat_params)))
        structured_module.lazy_load(params)

        # Save original input for backward
        x_saved = x

        # === Attention Block ===
        h = x
        normed1 = structured_module.input_layernorm(h)

        # Compute attention with softmax weights saved
        B, L, _ = normed1.shape
        attn = structured_module.self_attn

        q = attn.q_proj(normed1)
        k = attn.k_proj(normed1)
        v = attn.v_proj(normed1)

        # Reshape for multi-head attention
        q = q.reshape(B, L, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE (Qwen2 doesn't have Q/K norm)
        q_rope = attn.rope(q)
        k_rope = attn.rope(k)

        # GQA: repeat k, v
        n_rep = attn.n_heads // attn.n_kv_heads
        if n_rep > 1:
            k_expanded = mx.repeat(k_rope, n_rep, axis=1)
            v_expanded = mx.repeat(v, n_rep, axis=1)
        else:
            k_expanded = k_rope
            v_expanded = v

        # Scaled dot-product attention
        scores = (q_rope @ k_expanded.transpose(0, 1, 3, 2)) * attn.scale

        # Causal mask
        mask = mx.triu(mx.full((L, L), float('-inf')), k=1)
        scores = scores + mask

        # Softmax - save weights for backward
        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = attn_weights @ v_expanded

        # Reshape and output projection
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        attn_out = attn.o_proj(attn_output)

        # Residual connection
        h = h + attn_out
        h_after_attn = h

        # === MLP Block ===
        normed2 = structured_module.post_attention_layernorm(h)

        # MLP forward - save gate input for SiLU backward
        gate_out = structured_module.mlp.gate_proj(normed2)
        up_out = structured_module.mlp.up_proj(normed2)
        silu_out = nn.silu(gate_out)
        mlp_out = structured_module.mlp.down_proj(silu_out * up_out)

        # Residual connection
        y = h + mlp_out

        # Return output and intermediates needed for backward
        return (
            y.astype(dtype),
            x_saved.astype(dtype),  # For LoRA backward (layer input)
            normed1.astype(dtype),  # For attention LoRA backward
            attn_weights.astype(dtype),  # For softmax backward
            attn_output.astype(dtype),  # For O projection backward
            h_after_attn.astype(dtype),  # For MLP block input
            normed2.astype(dtype),  # For MLP LoRA backward
            gate_out.astype(dtype),  # For SiLU backward
            up_out.astype(dtype),  # For MLP backward
            silu_out.astype(dtype),  # For down_proj backward
        )

    shape = (batch_size, context_length, args.hidden_size)
    example_x = mx.random.normal(shape=shape, dtype=dtype)

    forward_fn_path = os.path.join(output_dir, f"{module_name}_forward_swift.mlxfn")
    f_args = (example_x,) + flatten_params
    mx.export_function(forward_fn_path, forward_with_intermediates, *f_args, shapeless=shapeless)

    # Save parameters
    trainable_params_dict = dict(zip(trainable_param_names, trainable_params))
    trainable_params_path = os.path.join(output_dir, f"trainable_{save_params_name}")
    save_params(trainable_params_path, trainable_params_dict, param_name_formatter)

    frozen_params_dict = dict(zip(frozen_param_names, frozen_params))
    frozen_params_path = os.path.join(output_dir, save_params_name)
    save_params(frozen_params_path, frozen_params_dict, param_name_formatter)

    x_name, y_name = variable_names["x"], variable_names["y"]
    formatted_param_names = [param_name_formatter(n) for n in flatten_param_names]

    # Output names include intermediates for Swift backward
    output_names = [
        y_name,
        f"{module_name}.x_saved",
        f"{module_name}.normed1",
        f"{module_name}.attn_weights",
        f"{module_name}.attn_output",
        f"{module_name}.h_after_attn",
        f"{module_name}.normed2",
        f"{module_name}.gate_out",
        f"{module_name}.up_out",
        f"{module_name}.silu_out",
    ]

    forward_run_config = RunFunctionConfig(
        function_name=f"{module_name}_forward_swift",
        input_names=[x_name] + formatted_param_names,
        output_names=output_names
    )

    return forward_run_config


def export_rope_backward(
    args: QLoRAModelArgs,
    output_dir: str,
    batch_size: int = 1,
    context_length: int = 256,
    shapeless: bool = False,
    module_name: str = "rope_backward",
) -> RunFunctionConfig:
    """
    Export RoPE backward as a utility function for Swift.

    RoPE backward is computed using VJP because RoPE is not orthogonal.
    This function allows Swift to call the correct RoPE backward.
    """
    # Create RoPE module
    rope = initialize_rope(
        args.head_dim,
        base=args.rope_theta,
        traditional=False,
        scaling_config=getattr(args, 'rope_scaling', None),
        max_position_embeddings=getattr(args, 'max_position_embeddings', 32768),
    )

    dtype = mx.bfloat16

    def rope_backward_fn(original_input: mx.array, grad_rope: mx.array) -> mx.array:
        """Compute RoPE backward using VJP."""
        def rope_fn(x):
            return rope(x)

        _, vjp_fn = mx.vjp(rope_fn, [original_input], [grad_rope])
        return vjp_fn[0].astype(dtype)

    # Example shapes for RoPE: [batch, heads, seq, head_dim]
    n_heads = args.num_attention_heads
    head_dim = args.head_dim
    example_shape = (batch_size, n_heads, context_length, head_dim)

    example_input = mx.random.normal(shape=example_shape, dtype=dtype)
    example_grad = mx.random.normal(shape=example_shape, dtype=dtype)

    fn_path = os.path.join(output_dir, f"{module_name}.mlxfn")
    mx.export_function(fn_path, rope_backward_fn, example_input, example_grad, shapeless=shapeless)

    return RunFunctionConfig(
        function_name=module_name,
        input_names=["rope_input", "rope_grad"],
        output_names=["rope_input_grad"]
    )
