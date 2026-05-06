#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
LoRA-Structured Backward for Qwen3 Transformer Block.

This module implements memory-efficient backward propagation for LoRA layers
by explicitly computing gradients using structured formulas instead of
relying on MLX's automatic differentiation.

Key insight from idea.md:
- Forward: h = Ax, Δy = Bh, y = W₀x + s·Δy
- Backward: grad_B = (s·g)hᵀ, grad_A = (Bᵀ(s·g))xᵀ, grad_x = W₀ᵀg + Aᵀ(Bᵀ(s·g))

Memory optimization: Store only x, recompute h = Ax during backward (Option A)
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
from .qwen3 import (
    QLoRAModelArgs,
    LazyLoadingAttention, LazyLoadingMLP,
)
from .runner import RunFunctionConfig
from .utils import save_params, get_grad_name


# =============================================================================
# Structured Backward Primitives (from idea.md)
# =============================================================================

def rms_norm_forward(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """RMSNorm forward pass."""
    norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return weight * (x / norm)


def rms_norm_backward(
    x: mx.array,
    weight: mx.array,
    grad_output: mx.array,
    eps: float = 1e-6
) -> mx.array:
    """
    RMSNorm backward pass - compute grad_x.

    For RMSNorm: y = weight * x / norm, where norm = sqrt(mean(x^2) + eps)

    Derivation:
        dy_i/dx_j = weight_i * (delta_ij / norm - x_i * x_j / (d * norm^3))

        grad_x_j = sum_i(grad_y_i * dy_i/dx_j)
                 = scaled_grad_j / norm - x_j * sum(scaled_grad * x) / (d * norm^3)
                 = scaled_grad_j / norm - x_j * mean(scaled_grad * x) / norm^3

    Args:
        x: Input to RMSNorm
        weight: RMSNorm weight
        grad_output: Gradient from upstream
        eps: Epsilon for numerical stability

    Returns:
        grad_x: Gradient w.r.t. input
    """
    norm_sq = mx.mean(x * x, axis=-1, keepdims=True) + eps
    norm = mx.sqrt(norm_sq)

    scaled_grad = grad_output * weight
    # Correct formula: divide by norm^3, NOT d * norm^3
    grad_x = scaled_grad / norm - x * mx.mean(scaled_grad * x, axis=-1, keepdims=True) / (norm_sq * norm)

    return grad_x


def lora_linear_forward(
    x: mx.array,
    w0_weight: mx.array,
    w0_scales: mx.array,
    w0_biases: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
    scale: float,
    group_size: int,
    bits: int,
) -> mx.array:
    """
    QLoRA forward pass: y = W₀x + s·(x @ A) @ B

    Args:
        x: Input (batch, seq, in_dim)
        w0_*: Quantized base weight parameters
        lora_a: LoRA A matrix (in_dim, rank)
        lora_b: LoRA B matrix (rank, out_dim)
        scale: LoRA scaling factor
        group_size, bits: Quantization parameters

    Returns:
        y: Output (batch, seq, out_dim)
    """
    # Base linear with quantized weights
    y_base = mx.quantized_matmul(
        x, w0_weight, w0_scales, w0_biases,
        transpose=True,
        group_size=group_size,
        bits=bits
    )

    # LoRA contribution: h = x @ A, delta_y = h @ B
    h = x @ lora_a
    delta_y = h @ lora_b

    return y_base + scale * delta_y


def lora_linear_backward(
    x: mx.array,
    w0_weight: mx.array,
    w0_scales: mx.array,
    w0_biases: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
    scale: float,
    group_size: int,
    bits: int,
    grad_output: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    LoRA-Structured Backward (from idea.md).

    Implements Option A: store only x, recompute h = Ax.

    Equations:
        grad_B = (s·g)ᵀ @ h = s · einsum('bso,bsr->ro', g, h)
        grad_h = (s·g) @ Bᵀ
        grad_A = xᵀ @ grad_h = einsum('bsi,bsr->ir', x, grad_h)
        grad_x = grad_h @ Aᵀ + g @ W₀

    Args:
        x: Input that was saved during forward
        w0_*: Quantized base weight parameters
        lora_a, lora_b: LoRA matrices
        scale: LoRA scaling factor
        group_size, bits: Quantization parameters
        grad_output: Gradient from upstream (batch, seq, out_dim)

    Returns:
        (grad_x, grad_a, grad_b)
    """
    # Recompute h (memory efficient - Option A from idea.md)
    h = x @ lora_a  # (batch, seq, rank)

    # Scaled gradient
    grad_scaled = scale * grad_output  # (batch, seq, out_dim)

    # grad_B = hᵀ @ (s·g) summed over batch and seq
    # h: (batch, seq, rank), grad_scaled: (batch, seq, out_dim)
    # grad_b: (rank, out_dim)
    grad_b = mx.einsum('bsr,bso->ro', h, grad_scaled)

    # grad_h = (s·g) @ Bᵀ
    grad_h = grad_scaled @ lora_b.T  # (batch, seq, rank)

    # grad_A = xᵀ @ grad_h summed over batch and seq
    # x: (batch, seq, in_dim), grad_h: (batch, seq, rank)
    # grad_a: (in_dim, rank)
    grad_a = mx.einsum('bsi,bsr->ir', x, grad_h)

    # grad_x from LoRA branch
    grad_x_lora = grad_h @ lora_a.T  # (batch, seq, in_dim)

    # grad_x from base branch: g @ W₀
    grad_x_base = mx.quantized_matmul(
        grad_output, w0_weight, w0_scales, w0_biases,
        transpose=False,
        group_size=group_size,
        bits=bits
    )

    grad_x = grad_x_lora + grad_x_base

    return grad_x, grad_a, grad_b


# =============================================================================
# Lazy Loading Modules for Structured Backward
# =============================================================================

class StructuredLoRALinear(nn.Module, LazyLoadingMixin):
    """LoRA Linear with explicit structured forward/backward."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        rank: int = 8,
        scale: float = 20.0,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.rank = rank
        self.scale = scale
        self.group_size = group_size
        self.bits = bits

        # Base quantized linear (frozen)
        self.linear = LazyLoadingQuantizedLinear(
            input_dims=input_dims,
            output_dims=output_dims,
            bias=bias,
            group_size=group_size,
            bits=bits
        )

        # LoRA matrices (will be loaded)
        self.lora_a = None
        self.lora_b = None

    def lazy_load(self, params: dict):
        self.linear.lazy_load(params["linear"])
        self.lora_a = params["lora_a"]
        self.lora_b = params["lora_b"]

    def forward(self, x: mx.array) -> mx.array:
        """Forward pass."""
        return lora_linear_forward(
            x,
            self.linear.weight, self.linear.scales, self.linear.biases,
            self.lora_a, self.lora_b,
            self.scale, self.group_size, self.bits
        )

    def backward(self, x: mx.array, grad_output: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Backward pass with structured gradients."""
        return lora_linear_backward(
            x,
            self.linear.weight, self.linear.scales, self.linear.biases,
            self.lora_a, self.lora_b,
            self.scale, self.group_size, self.bits,
            grad_output
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)

    @staticmethod
    def from_base(
        linear: LazyLoadingQuantizedLinear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        layer = StructuredLoRALinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            rank=r,
            scale=scale,
            bias=linear.bias is not None,
            group_size=linear.group_size,
            bits=linear.bits,
        )
        layer.linear = linear
        return layer


# =============================================================================
# Structured Transformer Block
# =============================================================================

class StructuredAttention(nn.Module, LazyLoadingMixin):
    """Attention with structured LoRA backward."""

    def __init__(self, args: QLoRAModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.args = args

        common_kwargs = {"bits": args.bits, "group_size": args.group_size, "bias": False}

        self.q_proj = StructuredLoRALinear(
            input_dims=dim, output_dims=self.n_heads * self.head_dim,
            rank=args.lora_args.rank, scale=args.lora_args.scale, **common_kwargs
        )
        self.k_proj = StructuredLoRALinear(
            input_dims=dim, output_dims=self.n_kv_heads * self.head_dim,
            rank=args.lora_args.rank, scale=args.lora_args.scale, **common_kwargs
        )
        self.v_proj = StructuredLoRALinear(
            input_dims=dim, output_dims=self.n_kv_heads * self.head_dim,
            rank=args.lora_args.rank, scale=args.lora_args.scale, **common_kwargs
        )
        self.o_proj = StructuredLoRALinear(
            input_dims=self.n_heads * self.head_dim, output_dims=dim,
            rank=args.lora_args.rank, scale=args.lora_args.scale, **common_kwargs
        )

        self.q_norm = RMSNorm(dims=self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(dims=self.head_dim, eps=args.rms_norm_eps)

        # RoPE (Rotary Position Embedding)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=getattr(args, 'rope_traditional', False),
            scaling_config=getattr(args, 'rope_scaling', None),
            max_position_embeddings=getattr(args, 'max_position_embeddings', 32768),
        )

    def lazy_load(self, params: dict):
        for name in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]:
            getattr(self, name).lazy_load(params[name])


class StructuredMLP(nn.Module, LazyLoadingMixin):
    """MLP with structured LoRA backward."""

    def __init__(self, args: QLoRAModelArgs):
        super().__init__()
        dim, hidden_dim = args.hidden_size, args.intermediate_size
        common_kwargs = {"bits": args.bits, "group_size": args.group_size, "bias": False}

        self.gate_proj = StructuredLoRALinear(
            input_dims=dim, output_dims=hidden_dim,
            rank=args.lora_args.rank, scale=args.lora_args.scale, **common_kwargs
        )
        self.up_proj = StructuredLoRALinear(
            input_dims=dim, output_dims=hidden_dim,
            rank=args.lora_args.rank, scale=args.lora_args.scale, **common_kwargs
        )
        self.down_proj = StructuredLoRALinear(
            input_dims=hidden_dim, output_dims=dim,
            rank=args.lora_args.rank, scale=args.lora_args.scale, **common_kwargs
        )

    def lazy_load(self, params: dict):
        for name in ["gate_proj", "up_proj", "down_proj"]:
            getattr(self, name).lazy_load(params[name])


class StructuredTransformerBlock(nn.Module, LazyLoadingMixin):
    """Transformer Block with structured LoRA backward."""

    def __init__(self, args: QLoRAModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self.self_attn = StructuredAttention(args)
        self.mlp = StructuredMLP(args)
        self.input_layernorm = RMSNorm(dims=args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(dims=args.hidden_size, eps=args.rms_norm_eps)

    def lazy_load(self, params: dict):
        for name in ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"]:
            getattr(self, name).lazy_load(params[name])


# =============================================================================
# Export Functions for Structured Backward
# =============================================================================

def export_structured_transformer_block(
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
    """
    Export transformer block with structured LoRA backward.

    The key difference from standard export:
    - Forward: Standard forward pass
    - Backward: Explicit gradient computation using structured formulas
               (NOT using mx.grad, implements idea.md directly)
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

    # Create structured module for forward/backward
    structured_module = StructuredTransformerBlock(args)

    # =========================================================================
    # Forward Function
    # =========================================================================
    def forward_fn(*inputs):
        x, *flat_params = inputs
        params = tree_unflatten(list(zip(flatten_param_names, flat_params)))
        structured_module.lazy_load(params)

        # Standard transformer block forward
        # x -> LayerNorm -> Attention -> + residual -> LayerNorm -> MLP -> + residual
        h = x

        # Attention block
        normed = structured_module.input_layernorm(h)
        attn_out = attention_forward(structured_module.self_attn, normed, args)
        h = h + attn_out

        # MLP block
        normed = structured_module.post_attention_layernorm(h)
        mlp_out = mlp_forward(structured_module.mlp, normed)
        h = h + mlp_out

        # Ensure output dtype matches input dtype (avoids int4 dequant → float32 issue)
        return h.astype(dtype)

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

    # =========================================================================
    # Backward Function (Structured - implements idea.md)
    # =========================================================================
    def backward_fn(*inputs):
        """
        Structured backward pass implementing idea.md.

        This function explicitly computes gradients for LoRA parameters
        using the structured formulas, achieving memory efficiency by
        recomputing intermediate values (h = Ax) instead of storing them.
        """
        x, dy, *flat_params = inputs
        params = tree_unflatten(list(zip(flatten_param_names, flat_params)))
        structured_module.lazy_load(params)

        # We need to recompute forward intermediates and then compute structured backward
        # This implements the full backward pass with explicit gradient computation

        grads = {}

        # =====================================================================
        # Recompute Forward Pass (storing intermediates for backward)
        # =====================================================================
        h = x

        # --- Attention Block Forward ---
        normed1 = structured_module.input_layernorm(h)
        attn_out, attn_intermediates = attention_forward_with_intermediates(
            structured_module.self_attn, normed1, args
        )
        h_after_attn = h + attn_out

        # --- MLP Block Forward ---
        normed2 = structured_module.post_attention_layernorm(h_after_attn)
        mlp_out, mlp_intermediates = mlp_forward_with_intermediates(
            structured_module.mlp, normed2
        )
        # h_final = h_after_attn + mlp_out (this equals y)

        # =====================================================================
        # Backward Pass (Structured)
        # =====================================================================
        grad_h = dy  # Gradient from upstream

        # --- MLP Block Backward ---
        grad_mlp_out = grad_h  # Due to residual connection
        grad_normed2, mlp_grads = mlp_backward_structured(
            structured_module.mlp, normed2, mlp_intermediates, grad_mlp_out
        )
        grads.update(mlp_grads)

        # Backward through post_attention_layernorm
        grad_h_after_attn = grad_h + rms_norm_backward(
            h_after_attn,
            structured_module.post_attention_layernorm.weight,
            grad_normed2,
            args.rms_norm_eps
        )

        # --- Attention Block Backward ---
        grad_attn_out = grad_h_after_attn  # Due to residual connection
        grad_normed1, attn_grads = attention_backward_structured(
            structured_module.self_attn, normed1, attn_intermediates, grad_attn_out, args
        )
        grads.update(attn_grads)

        # Backward through input_layernorm
        grad_x = grad_h_after_attn + rms_norm_backward(
            x,
            structured_module.input_layernorm.weight,
            grad_normed1,
            args.rms_norm_eps
        )

        # Return gradients in the expected order
        # Ensure gradient dtypes match input dtype
        trainable_grads = tuple(grads.get(name, mx.zeros_like(p)).astype(dtype)
                                for name, p in zip(trainable_param_names, trainable_params))

        return (grad_x.astype(dtype),) + trainable_grads

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


# =============================================================================
# Forward/Backward Helper Functions
# =============================================================================

def attention_forward(attn: StructuredAttention, x: mx.array, args: QLoRAModelArgs) -> mx.array:
    """Standard attention forward with QK norm and RoPE."""
    B, L, _ = x.shape

    q = attn.q_proj(x)
    k = attn.k_proj(x)
    v = attn.v_proj(x)

    # Reshape for multi-head attention
    q = q.reshape(B, L, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

    # Apply QK norm
    q = attn.q_norm(q)
    k = attn.k_norm(k)

    # Apply RoPE (Rotary Position Embedding)
    q = attn.rope(q)
    k = attn.rope(k)

    # GQA: repeat k, v for grouped query attention
    n_rep = attn.n_heads // attn.n_kv_heads
    if n_rep > 1:
        k = mx.repeat(k, n_rep, axis=1)
        v = mx.repeat(v, n_rep, axis=1)

    # Scaled dot-product attention with causal mask
    scores = (q @ k.transpose(0, 1, 3, 2)) * attn.scale

    # Causal mask
    mask = mx.triu(mx.full((L, L), float('-inf')), k=1)
    scores = scores + mask

    attn_weights = mx.softmax(scores, axis=-1)
    attn_output = attn_weights @ v

    # Reshape back
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)

    # Output projection
    output = attn.o_proj(attn_output)

    return output


def rope_backward(grad_rope: mx.array, rope_module, original_input: mx.array) -> mx.array:
    """
    Backward pass through RoPE using VJP.

    RoPE is NOT orthogonal in general, so we must use the correct VJP.
    """
    def rope_fn(x):
        return rope_module(x)

    _, vjp_fn = mx.vjp(rope_fn, [original_input], [grad_rope])
    return vjp_fn[0]


def attention_forward_with_intermediates(
    attn: StructuredAttention, x: mx.array, args: QLoRAModelArgs
) -> tuple[mx.array, dict]:
    """Attention forward that saves intermediates for backward (with QK norm and RoPE)."""
    B, L, _ = x.shape

    q = attn.q_proj(x)
    k = attn.k_proj(x)
    v = attn.v_proj(x)

    q_reshaped = q.reshape(B, L, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
    k_reshaped = k.reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    v_reshaped = v.reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

    # Apply QK norm
    q_normed = attn.q_norm(q_reshaped)
    k_normed = attn.k_norm(k_reshaped)

    # Apply RoPE (Rotary Position Embedding)
    q_rope = attn.rope(q_normed)
    k_rope = attn.rope(k_normed)

    n_rep = attn.n_heads // attn.n_kv_heads
    if n_rep > 1:
        k_expanded = mx.repeat(k_rope, n_rep, axis=1)
        v_expanded = mx.repeat(v_reshaped, n_rep, axis=1)
    else:
        k_expanded = k_rope
        v_expanded = v_reshaped

    scores = (q_rope @ k_expanded.transpose(0, 1, 3, 2)) * attn.scale
    mask = mx.triu(mx.full((L, L), float('-inf')), k=1)
    scores = scores + mask

    attn_weights = mx.softmax(scores, axis=-1)
    attn_output = attn_weights @ v_expanded

    attn_output_reshaped = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    output = attn.o_proj(attn_output_reshaped)

    intermediates = {
        'x': x,
        'q': q, 'k': k, 'v': v,
        'q_reshaped': q_reshaped, 'k_reshaped': k_reshaped, 'v_reshaped': v_reshaped,
        'q_normed': q_normed, 'k_normed': k_normed,
        'q_rope': q_rope, 'k_rope': k_rope,
        'k_expanded': k_expanded, 'v_expanded': v_expanded,
        'attn_weights': attn_weights,
        'attn_output': attn_output,
        'attn_output_reshaped': attn_output_reshaped,
    }

    return output, intermediates


def attention_backward_structured(
    attn: StructuredAttention,
    x: mx.array,
    intermediates: dict,
    grad_output: mx.array,
    args: QLoRAModelArgs,
) -> tuple[mx.array, dict]:
    """
    Structured backward for attention using idea.md formulas (with QK norm and RoPE).
    """
    B, L, _ = x.shape
    grads = {}

    # O projection backward (structured LoRA backward)
    attn_output_reshaped = intermediates['attn_output_reshaped']
    grad_attn_output_reshaped, grad_o_a, grad_o_b = attn.o_proj.backward(
        attn_output_reshaped, grad_output
    )
    grads['self_attn.o_proj.lora_a'] = grad_o_a
    grads['self_attn.o_proj.lora_b'] = grad_o_b

    # Reshape gradient back through attention output
    grad_attn_output = grad_attn_output_reshaped.reshape(B, L, attn.n_heads, attn.head_dim)
    grad_attn_output = grad_attn_output.transpose(0, 2, 1, 3)

    # Attention backward - use RoPE-transformed q and k
    attn_weights = intermediates['attn_weights']
    v_expanded = intermediates['v_expanded']
    k_expanded = intermediates['k_expanded']  # This is k_rope expanded
    q_rope = intermediates['q_rope']

    # grad_v = attn_weights.T @ grad_attn_output
    grad_v_expanded = attn_weights.transpose(0, 1, 3, 2) @ grad_attn_output

    # grad_attn_weights = grad_attn_output @ v.T
    grad_attn_weights = grad_attn_output @ v_expanded.transpose(0, 1, 3, 2)

    # Softmax backward
    grad_scores = attn_weights * (grad_attn_weights - mx.sum(grad_attn_weights * attn_weights, axis=-1, keepdims=True))
    grad_scores = grad_scores * attn.scale

    # grad_q_rope = grad_scores @ k_expanded (k_expanded is already after RoPE)
    grad_q_rope = grad_scores @ k_expanded

    # grad_k_expanded = grad_scores.T @ q_rope
    grad_k_expanded = grad_scores.transpose(0, 1, 3, 2) @ q_rope

    # Handle GQA: sum gradients for repeated k, v
    n_rep = attn.n_heads // attn.n_kv_heads
    if n_rep > 1:
        grad_k_rope = grad_k_expanded.reshape(B, attn.n_kv_heads, n_rep, L, attn.head_dim).sum(axis=2)
        grad_v_reshaped = grad_v_expanded.reshape(B, attn.n_kv_heads, n_rep, L, attn.head_dim).sum(axis=2)
    else:
        grad_k_rope = grad_k_expanded
        grad_v_reshaped = grad_v_expanded

    # Backward through RoPE
    q_normed = intermediates['q_normed']
    k_normed = intermediates['k_normed']
    grad_q_normed = rope_backward(grad_q_rope, attn.rope, q_normed)
    grad_k_normed = rope_backward(grad_k_rope, attn.rope, k_normed)

    # QK norm backward
    q_reshaped = intermediates['q_reshaped']
    k_reshaped = intermediates['k_reshaped']

    grad_q_reshaped = rms_norm_backward(q_reshaped, attn.q_norm.weight, grad_q_normed, args.rms_norm_eps)
    grad_k_reshaped = rms_norm_backward(k_reshaped, attn.k_norm.weight, grad_k_normed, args.rms_norm_eps)

    # Reshape gradients back
    grad_q = grad_q_reshaped.transpose(0, 2, 1, 3).reshape(B, L, -1)
    grad_k = grad_k_reshaped.transpose(0, 2, 1, 3).reshape(B, L, -1)
    grad_v = grad_v_reshaped.transpose(0, 2, 1, 3).reshape(B, L, -1)

    # Q, K, V projection backward (structured LoRA backward)
    grad_x_q, grad_q_a, grad_q_b = attn.q_proj.backward(x, grad_q)
    grad_x_k, grad_k_a, grad_k_b = attn.k_proj.backward(x, grad_k)
    grad_x_v, grad_v_a, grad_v_b = attn.v_proj.backward(x, grad_v)

    grads['self_attn.q_proj.lora_a'] = grad_q_a
    grads['self_attn.q_proj.lora_b'] = grad_q_b
    grads['self_attn.k_proj.lora_a'] = grad_k_a
    grads['self_attn.k_proj.lora_b'] = grad_k_b
    grads['self_attn.v_proj.lora_a'] = grad_v_a
    grads['self_attn.v_proj.lora_b'] = grad_v_b

    grad_x = grad_x_q + grad_x_k + grad_x_v

    return grad_x, grads


def mlp_forward(mlp: StructuredMLP, x: mx.array) -> mx.array:
    """Standard MLP forward: down(silu(gate(x)) * up(x))."""
    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    hidden = nn.silu(gate) * up
    return mlp.down_proj(hidden)


def mlp_forward_with_intermediates(mlp: StructuredMLP, x: mx.array) -> tuple[mx.array, dict]:
    """MLP forward that saves intermediates for backward."""
    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    gate_silu = nn.silu(gate)
    hidden = gate_silu * up
    output = mlp.down_proj(hidden)

    intermediates = {
        'x': x,
        'gate': gate,
        'up': up,
        'gate_silu': gate_silu,
        'hidden': hidden,
    }

    return output, intermediates


def mlp_backward_structured(
    mlp: StructuredMLP,
    x: mx.array,
    intermediates: dict,
    grad_output: mx.array,
) -> tuple[mx.array, dict]:
    """
    Structured backward for MLP using idea.md formulas.
    """
    grads = {}

    hidden = intermediates['hidden']
    gate = intermediates['gate']
    up = intermediates['up']
    gate_silu = intermediates['gate_silu']

    # Down projection backward
    grad_hidden, grad_down_a, grad_down_b = mlp.down_proj.backward(hidden, grad_output)
    grads['mlp.down_proj.lora_a'] = grad_down_a
    grads['mlp.down_proj.lora_b'] = grad_down_b

    # hidden = silu(gate) * up
    # grad_gate_silu = grad_hidden * up
    # grad_up = grad_hidden * gate_silu
    grad_gate_silu = grad_hidden * up
    grad_up = grad_hidden * gate_silu

    # SiLU backward: silu(x) = x * sigmoid(x)
    # d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    #              = silu(x) / x + silu(x) * (1 - sigmoid(x))
    # Simpler: d/dx silu(x) = silu(x) + sigmoid(x) * (1 - silu(x) / x) when x != 0
    # Most stable: use autograd identity
    sigmoid_gate = mx.sigmoid(gate)
    grad_gate = grad_gate_silu * (sigmoid_gate + gate * sigmoid_gate * (1 - sigmoid_gate))

    # Gate and Up projection backward
    grad_x_gate, grad_gate_a, grad_gate_b = mlp.gate_proj.backward(x, grad_gate)
    grad_x_up, grad_up_a, grad_up_b = mlp.up_proj.backward(x, grad_up)

    grads['mlp.gate_proj.lora_a'] = grad_gate_a
    grads['mlp.gate_proj.lora_b'] = grad_gate_b
    grads['mlp.up_proj.lora_a'] = grad_up_a
    grads['mlp.up_proj.lora_b'] = grad_up_b

    grad_x = grad_x_gate + grad_x_up

    return grad_x, grads
