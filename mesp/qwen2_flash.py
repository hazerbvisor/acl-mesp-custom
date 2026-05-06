#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
LoRA Training with Flash Attention (Memory-Efficient SDPA) for Qwen2.

Key differences from Qwen3:
- Qwen2 uses bias=True for Q, K, V projections
- Qwen2 doesn't have QK normalization
- head_dim = hidden_size // num_attention_heads
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
from .qwen2 import QLoRAModelArgs
from .lora import lazy_loading_linear_to_lora
from .runner import RunFunctionConfig
from .utils import save_params, get_grad_name, product_sum


# =============================================================================
# Flash Attention Module for Qwen2
# =============================================================================

class FlashAttention(nn.Module, LazyLoadingMixin):
    """Attention using mx.fast.scaled_dot_product_attention for memory efficiency."""

    def __init__(self, args: QLoRAModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        # Qwen2: head_dim = hidden_size // num_attention_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.args = args

        # Qwen2 uses bias=True for Q, K, V projections
        common_kwargs = {"bits": args.bits, "group_size": args.group_size, "bias": True}

        self.q_proj = LazyLoadingQuantizedLinear(
            input_dims=dim, output_dims=self.n_heads * self.head_dim, **common_kwargs
        )
        self.k_proj = LazyLoadingQuantizedLinear(
            input_dims=dim, output_dims=self.n_kv_heads * self.head_dim, **common_kwargs
        )
        self.v_proj = LazyLoadingQuantizedLinear(
            input_dims=dim, output_dims=self.n_kv_heads * self.head_dim, **common_kwargs
        )
        # O projection has no bias
        self.o_proj = LazyLoadingQuantizedLinear(
            input_dims=self.n_heads * self.head_dim, output_dims=dim,
            bits=args.bits, group_size=args.group_size, bias=False
        )

        # Qwen2 doesn't have QK norm, only RoPE
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def lazy_load(self, params: dict):
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            getattr(self, name).lazy_load(params[name])

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass using Flash Attention (SDPA)."""
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (B, L, D) -> (B, N, L, head_dim)
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE (no QK norm in Qwen2)
        q = self.rope(q)
        k = self.rope(k)

        # Flash Attention: Memory-efficient SDPA
        attn_output = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            mask="causal"
        )

        # Reshape back: (B, N, L, head_dim) -> (B, L, D)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(attn_output)


class FlashMLP(nn.Module, LazyLoadingMixin):
    """MLP module."""

    def __init__(self, args: QLoRAModelArgs):
        super().__init__()
        dim, hidden_dim = args.hidden_size, args.intermediate_size
        common_kwargs = {"bits": args.bits, "group_size": args.group_size, "bias": False}

        self.gate_proj = LazyLoadingQuantizedLinear(input_dims=dim, output_dims=hidden_dim, **common_kwargs)
        self.up_proj = LazyLoadingQuantizedLinear(input_dims=dim, output_dims=hidden_dim, **common_kwargs)
        self.down_proj = LazyLoadingQuantizedLinear(input_dims=hidden_dim, output_dims=dim, **common_kwargs)

    def lazy_load(self, params: dict):
        for name in ["gate_proj", "up_proj", "down_proj"]:
            getattr(self, name).lazy_load(params[name])

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class FlashTransformerBlock(nn.Module, LazyLoadingMixin):
    """Transformer Block with Flash Attention for Qwen2."""

    def __init__(self, args: QLoRAModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self.self_attn = FlashAttention(args)
        self.mlp = FlashMLP(args)
        self.input_layernorm = RMSNorm(dims=args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(dims=args.hidden_size, eps=args.rms_norm_eps)

    def lazy_load(self, params: dict):
        for name in ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"]:
            getattr(self, name).lazy_load(params[name])

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


# =============================================================================
# Export Functions
# =============================================================================

def export_flash_transformer_block(
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
    forward_only: bool = False,
) -> tuple[RunFunctionConfig, RunFunctionConfig | None]:
    """Export transformer block with Flash Attention for Qwen2."""
    assert "x" in variable_names and "y" in variable_names

    # Get parameters
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

    argnames = ("x",) + trainable_param_names

    # Create Flash Attention module
    flash_module = FlashTransformerBlock(args)
    from .lora import LoRAArgs
    export_lora_args = LoRAArgs(
        rank=args.lora_args.rank,
        dropout=args.lora_args.dropout,
        scale=args.lora_args.scale,
        use_structured_backward=False,
        target_modules=args.lora_args.target_modules,
    )
    lazy_loading_linear_to_lora(flash_module, export_lora_args)

    # Forward Function
    def forward_fn(*inputs):
        x, *flat_params = inputs
        params = tree_unflatten(list(zip(flatten_param_names, flat_params)))
        flash_module.lazy_load(params)
        return flash_module(x).astype(dtype)

    shape = (batch_size, context_length, args.hidden_size)
    example_x = mx.random.normal(shape=shape, dtype=dtype)

    forward_fn_path = os.path.join(output_dir, f"{module_name}_forward.mlxfn")
    f_args = (example_x,) + flatten_params
    mx.export_function(forward_fn_path, forward_fn, *f_args, shapeless=shapeless)

    # Save parameters
    trainable_params_dict = dict(zip(trainable_param_names, trainable_params))
    trainable_params_path = os.path.join(output_dir, f"trainable_{save_params_name}")
    save_params(trainable_params_path, trainable_params_dict, param_name_formatter)

    frozen_params_dict = dict(zip(frozen_param_names, frozen_params))
    frozen_params_path = os.path.join(output_dir, save_params_name)
    save_params(frozen_params_path, frozen_params_dict, param_name_formatter)

    x_name, y_name = variable_names["x"], variable_names["y"]
    formatted_param_names = [param_name_formatter(n) for n in flatten_param_names]

    forward_run_config = RunFunctionConfig(
        function_name=f"{module_name}_forward",
        input_names=[x_name] + formatted_param_names,
        output_names=[y_name]
    )

    if forward_only:
        return forward_run_config, None

    # Backward Function
    def loss_fn(**kwargs):
        x, dy = kwargs.pop("x"), kwargs.pop("dy")
        params = tree_unflatten(list(zip(kwargs.keys(), kwargs.values())))
        flash_module.lazy_load(params)
        return product_sum(flash_module(x).astype(dtype), dy)

    def backward_fn(*inputs):
        x, dy, *flat_params = inputs
        kwargs = dict(zip(flatten_param_names, flat_params))
        _, grads = mx.grad(loss_fn, argnames=argnames)(**kwargs, x=x, dy=dy)
        return tuple(grads[name].astype(dtype) for name in argnames)

    example_dy = mx.random.normal(shape=shape, dtype=dtype)
    backward_fn_path = os.path.join(output_dir, f"{module_name}_backward.mlxfn")
    b_args = (example_x, example_dy) + flatten_params
    mx.export_function(backward_fn_path, backward_fn, *b_args, shapeless=shapeless)

    backward_run_config = RunFunctionConfig(
        function_name=f"{module_name}_backward",
        input_names=[x_name, get_grad_name(y_name)] + formatted_param_names,
        output_names=[get_grad_name(x_name)] + [
            get_grad_name(param_name_formatter(n)) for n in trainable_param_names
        ]
    )

    return forward_run_config, backward_run_config
