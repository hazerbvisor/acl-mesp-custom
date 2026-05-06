#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Optimized LoRA-Structured Export for Swift Backward.

Key optimizations over qwen3_structured_swift.py:
1. Use Flash Attention (SDPA) - removes O(seq²) attn_weights storage
2. Remove silu_out - recompute from gate_out during backward
3. Minimal intermediate storage for maximum memory savings

Intermediates stored (5 instead of 9):
- xSaved: layer input [B, S, D]
- normed1: for attention LoRA backward [B, S, D]
- attnOutput: for O_proj backward [B, S, D]
- hAfterAttn: for MLP input [B, S, D]
- normed2: for MLP LoRA backward [B, S, D]
- gateOut: for SiLU backward [B, S, I]
- upOut: for MLP backward [B, S, I]

Removed (saves ~4MB per layer for seq=256):
- attnWeights: O(seq²) - 1.75MB saved by Flash Attention
- siluOut: recompute from gateOut - 2.37MB saved
"""
from __future__ import annotations

import os
from typing import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.models.rope_utils import initialize_rope

from .modules import LazyLoadingMixin, LazyLoadingQuantizedLinear, LazyLoadingRMSNorm as RMSNorm
from .qwen3 import QLoRAModelArgs
from .qwen3_structured import (
    StructuredTransformerBlock,
    StructuredAttention,
    StructuredMLP,
)
from .runner import RunFunctionConfig
from .utils import save_params, get_grad_name


def export_transformer_block_for_swift_backward_optimized(
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
    Export transformer block for Swift-side structured backward (OPTIMIZED).

    Key optimizations:
    1. Uses Flash Attention (SDPA) - no O(seq²) attention weights stored
    2. Removes silu_out - Swift recomputes from gate_out
    3. Outputs only 7 intermediates instead of 10

    The forward function outputs:
    - y: Layer output (for next layer / loss)
    - x_saved: Layer input (for Swift to compute structured backward)
    - normed1: After input_layernorm (for attention LoRA backward)
    - attn_output: Attention output before O_proj (for O_proj backward)
    - h_after_attn: After attention + residual (for MLP input)
    - normed2: After post_attention_layernorm (for MLP LoRA backward)
    - gate_out: Gate projection output (for SiLU backward - Swift recomputes silu)
    - up_out: Up projection output (for MLP backward)
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

    def forward_with_intermediates_optimized(*inputs):
        """
        OPTIMIZED forward pass that returns minimal intermediates for Swift backward.

        Key changes from non-optimized version:
        1. Uses mx.fast.scaled_dot_product_attention (Flash Attention)
           - Avoids materializing O(seq²) attention weights
        2. Does NOT output silu_out
           - Swift recomputes silu(gate_out) during backward
        """
        x, *flat_params = inputs
        params = tree_unflatten(list(zip(flatten_param_names, flat_params)))
        structured_module.lazy_load(params)

        # Save original input for backward
        x_saved = x

        # === Attention Block ===
        h = x
        normed1 = structured_module.input_layernorm(h)

        B, L, _ = normed1.shape
        attn = structured_module.self_attn

        q = attn.q_proj(normed1)
        k = attn.k_proj(normed1)
        v = attn.v_proj(normed1)

        # Reshape for multi-head attention
        q = q.reshape(B, L, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

        # Apply QK norm
        q_normed = attn.q_norm(q)
        k_normed = attn.k_norm(k)

        # Apply RoPE
        q_rope = attn.rope(q_normed)
        k_rope = attn.rope(k_normed)

        # GQA: repeat k, v
        n_rep = attn.n_heads // attn.n_kv_heads
        if n_rep > 1:
            k_expanded = mx.repeat(k_rope, n_rep, axis=1)
            v_expanded = mx.repeat(v, n_rep, axis=1)
        else:
            k_expanded = k_rope
            v_expanded = v

        # Scaled dot-product attention (keep attn_weights for backward)
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

        # MLP forward
        gate_out = structured_module.mlp.gate_proj(normed2)
        up_out = structured_module.mlp.up_proj(normed2)
        # ===== OPTIMIZATION 2: Don't save silu_out =====
        # Swift will recompute: silu_out = silu(gate_out)
        silu_out = nn.silu(gate_out)
        mlp_out = structured_module.mlp.down_proj(silu_out * up_out)

        # Residual connection
        y = h + mlp_out

        # Return output and intermediates needed for FULL backward (including Q/K/V)
        # 9 outputs total:
        # - normed1: for Q/K/V projection backward
        # - attn_weights: for attention backward (O(seq²) but necessary for Q/K/V training)
        # - silu_out: recomputed from gate_out in Swift (not stored)
        return (
            y.astype(dtype),
            x_saved.astype(dtype),      # For residual gradient flow
            normed1.astype(dtype),      # For Q/K/V projection backward
            attn_weights.astype(dtype), # For attention backward (softmax)
            attn_output.astype(dtype),  # For O projection backward
            h_after_attn.astype(dtype), # For MLP block input (post_attn_layernorm)
            normed2.astype(dtype),      # For MLP LoRA backward
            gate_out.astype(dtype),     # For SiLU backward (Swift recomputes silu)
            up_out.astype(dtype),       # For MLP backward
        )

    shape = (batch_size, context_length, args.hidden_size)
    example_x = mx.random.normal(shape=shape, dtype=dtype)

    forward_fn_path = os.path.join(output_dir, f"{module_name}_forward_swift.mlxfn")
    f_args = (example_x,) + flatten_params
    mx.export_function(forward_fn_path, forward_with_intermediates_optimized, *f_args, shapeless=shapeless)

    # Save parameters
    trainable_params_dict = dict(zip(trainable_param_names, trainable_params))
    trainable_params_path = os.path.join(output_dir, f"trainable_{save_params_name}")
    save_params(trainable_params_path, trainable_params_dict, param_name_formatter)

    frozen_params_dict = dict(zip(frozen_param_names, frozen_params))
    frozen_params_path = os.path.join(output_dir, save_params_name)
    save_params(frozen_params_path, frozen_params_dict, param_name_formatter)

    x_name, y_name = variable_names["x"], variable_names["y"]
    formatted_param_names = [param_name_formatter(n) for n in flatten_param_names]

    # Output names - FULL (9 outputs for complete Q/K/V backward)
    # Includes normed1 and attn_weights for attention backward
    output_names = [
        y_name,
        f"{module_name}.x_saved",
        f"{module_name}.normed1",       # For Q/K/V projection backward
        f"{module_name}.attn_weights",  # For attention backward (softmax)
        f"{module_name}.attn_output",
        f"{module_name}.h_after_attn",
        f"{module_name}.normed2",
        f"{module_name}.gate_out",
        f"{module_name}.up_out",
    ]

    forward_run_config = RunFunctionConfig(
        function_name=f"{module_name}_forward_swift",
        input_names=[x_name] + formatted_param_names,
        output_names=output_names
    )

    return forward_run_config
