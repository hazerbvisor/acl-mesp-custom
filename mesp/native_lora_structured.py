#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Native MLX LoRA-Structured Implementation.

Uses @mx.custom_function + @vjp for memory-efficient backward propagation
WITHOUT exporting to .mlxfn files. This allows MLX to optimize the computation
graph directly.

Key benefits over export-based approach:
1. Native MLX graph optimization
2. No intermediate storage overhead from export/import
3. Direct use of custom backward kernels
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


# =============================================================================
# Custom Function: LoRA QLinear with Structured Backward
# =============================================================================

@mx.custom_function
def lora_qlinear_forward(
    x: mx.array,
    w0_weight: mx.array,
    w0_scales: mx.array,
    w0_biases: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
    scale: mx.array,
    group_size: mx.array,
    bits: mx.array,
) -> mx.array:
    """
    QLoRA forward: y = W0*x + scale * (x @ A) @ B

    Uses quantized base weight and trainable LoRA matrices.
    """
    # Base linear with quantized weights
    y_base = mx.quantized_matmul(
        x, w0_weight, w0_scales, w0_biases,
        transpose=True,
        group_size=int(group_size.item()),
        bits=int(bits.item())
    )

    # LoRA: h = x @ A, delta_y = h @ B
    h = x @ lora_a
    delta_y = h @ lora_b

    return y_base + scale * delta_y


@lora_qlinear_forward.vjp
def lora_qlinear_backward(
    primals: tuple,
    cotangent: mx.array,
    outputs: tuple,
) -> tuple:
    """
    Structured backward for QLoRA.

    Key optimization: Recompute h = x @ A instead of storing it.
    This saves memory proportional to (batch * seq * rank).

    Gradients computed:
    - grad_x = grad_output @ W0 + (grad_output @ B.T @ scale) @ A.T
    - grad_A = x.T @ (scale * grad_output @ B.T)
    - grad_B = (x @ A).T @ (scale * grad_output)
    """
    x, w0_weight, w0_scales, w0_biases, lora_a, lora_b, scale, group_size, bits = primals
    grad_output = cotangent

    gs = int(group_size.item())
    b = int(bits.item())

    # Scaled gradient for LoRA branch
    grad_scaled = scale * grad_output

    # Recompute h (memory efficient - don't store during forward)
    h = x @ lora_a  # (batch, seq, rank)

    # grad_B = h.T @ grad_scaled (summed over batch, seq)
    grad_b = mx.einsum('bsr,bso->ro', h, grad_scaled)

    # grad_h = grad_scaled @ B.T
    grad_h = grad_scaled @ lora_b.T  # (batch, seq, rank)

    # grad_A = x.T @ grad_h (summed over batch, seq)
    grad_a = mx.einsum('bsi,bsr->ir', x, grad_h)

    # grad_x from LoRA branch
    grad_x_lora = grad_h @ lora_a.T

    # grad_x from base branch: grad_output @ W0
    grad_x_base = mx.quantized_matmul(
        grad_output, w0_weight, w0_scales, w0_biases,
        transpose=False,
        group_size=gs,
        bits=b
    )

    grad_x = grad_x_lora + grad_x_base

    # grad_scale (for completeness)
    delta_y_unscaled = h @ lora_b
    grad_scale = mx.sum(grad_output * delta_y_unscaled)

    # Return gradients for all primals
    # Frozen weights get zero gradients
    return (
        grad_x,
        mx.zeros_like(w0_weight),  # w0_weight (frozen)
        mx.zeros_like(w0_scales),  # w0_scales (frozen)
        mx.zeros_like(w0_biases),  # w0_biases (frozen)
        grad_a,
        grad_b,
        grad_scale,
        mx.array(0.0),  # group_size
        mx.array(0.0),  # bits
    )


# =============================================================================
# LoRA Linear Layer with Structured Backward
# =============================================================================

class StructuredLoRALinear(nn.Module):
    """
    QLoRA Linear layer using @mx.custom_function for memory-efficient backward.

    Forward: y = W0*x + scale * (x @ A) @ B
    Backward: Uses structured gradient computation, recomputes h = x @ A
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.group_size = group_size
        self.bits = bits

        # Base quantized linear (frozen)
        self.linear = nn.QuantizedLinear(
            input_dims, output_dims, bias=bias,
            group_size=group_size, bits=bits
        )
        self.linear.freeze()

        # LoRA matrices (trainable)
        # A: (in_features, rank) - initialized with small random values
        # B: (rank, out_features) - initialized to zero
        lora_scale = 1.0 / math.sqrt(rank)
        self.lora_a = mx.random.normal(shape=(input_dims, rank)) * lora_scale
        self.lora_b = mx.zeros((rank, output_dims))

    def __call__(self, x: mx.array) -> mx.array:
        return lora_qlinear_forward(
            x,
            self.linear.weight,
            self.linear.scales,
            self.linear.biases,
            self.lora_a,
            self.lora_b,
            mx.array(self.scale),
            mx.array(self.group_size),
            mx.array(self.bits),
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.QuantizedLinear,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> "StructuredLoRALinear":
        """Create LoRA layer from existing quantized linear."""
        layer = cls(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            group_size=linear.group_size,
            bits=linear.bits,
        )
        # Copy base weights
        layer.linear.weight = linear.weight
        layer.linear.scales = linear.scales
        layer.linear.biases = linear.biases
        if linear.bias is not None:
            layer.linear.bias = linear.bias
        layer.linear.freeze()
        return layer


# =============================================================================
# Standard LoRA Linear (for comparison)
# =============================================================================

class StandardLoRALinear(nn.Module):
    """
    Standard QLoRA Linear layer using MLX's automatic differentiation.
    No custom backward - uses standard autograd.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.group_size = group_size
        self.bits = bits

        # Base quantized linear (frozen)
        self.linear = nn.QuantizedLinear(
            input_dims, output_dims, bias=bias,
            group_size=group_size, bits=bits
        )
        self.linear.freeze()

        # LoRA matrices (trainable)
        lora_scale = 1.0 / math.sqrt(rank)
        self.lora_a = mx.random.normal(shape=(input_dims, rank)) * lora_scale
        self.lora_b = mx.zeros((rank, output_dims))

    def __call__(self, x: mx.array) -> mx.array:
        # Base forward
        y_base = self.linear(x)

        # LoRA forward (standard, MLX will build autograd graph)
        h = x @ self.lora_a
        delta_y = h @ self.lora_b

        return y_base + self.scale * delta_y

    @classmethod
    def from_linear(
        cls,
        linear: nn.QuantizedLinear,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> "StandardLoRALinear":
        """Create LoRA layer from existing quantized linear."""
        layer = cls(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            group_size=linear.group_size,
            bits=linear.bits,
        )
        layer.linear.weight = linear.weight
        layer.linear.scales = linear.scales
        layer.linear.biases = linear.biases
        if linear.bias is not None:
            layer.linear.bias = linear.bias
        layer.linear.freeze()
        return layer


# =============================================================================
# Model Args
# =============================================================================

@dataclass
class LoRAModelArgs:
    hidden_size: int = 896
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 4864
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    # Quantization
    group_size: int = 64
    bits: int = 4
    # LoRA
    lora_rank: int = 8
    lora_alpha: float = 16.0


# =============================================================================
# Attention with Structured LoRA
# =============================================================================

class StructuredAttention(nn.Module):
    """Multi-head attention with structured LoRA on Q, K, V, O projections."""

    def __init__(self, args: LoRAModelArgs, use_structured: bool = True):
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5

        LoRAClass = StructuredLoRALinear if use_structured else StandardLoRALinear

        self.q_proj = LoRAClass(
            args.hidden_size, self.n_heads * self.head_dim,
            rank=args.lora_rank, alpha=args.lora_alpha,
            group_size=args.group_size, bits=args.bits
        )
        self.k_proj = LoRAClass(
            args.hidden_size, self.n_kv_heads * self.head_dim,
            rank=args.lora_rank, alpha=args.lora_alpha,
            group_size=args.group_size, bits=args.bits
        )
        self.v_proj = LoRAClass(
            args.hidden_size, self.n_kv_heads * self.head_dim,
            rank=args.lora_rank, alpha=args.lora_alpha,
            group_size=args.group_size, bits=args.bits
        )
        self.o_proj = LoRAClass(
            self.n_heads * self.head_dim, args.hidden_size,
            rank=args.lora_rank, alpha=args.lora_alpha,
            group_size=args.group_size, bits=args.bits
        )

        # RoPE
        self.rope = nn.RoPE(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # GQA: repeat k, v
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = attn_weights @ v

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(attn_output)


# =============================================================================
# MLP with Structured LoRA
# =============================================================================

class StructuredMLP(nn.Module):
    """MLP with structured LoRA on gate, up, down projections."""

    def __init__(self, args: LoRAModelArgs, use_structured: bool = True):
        super().__init__()

        LoRAClass = StructuredLoRALinear if use_structured else StandardLoRALinear

        self.gate_proj = LoRAClass(
            args.hidden_size, args.intermediate_size,
            rank=args.lora_rank, alpha=args.lora_alpha,
            group_size=args.group_size, bits=args.bits
        )
        self.up_proj = LoRAClass(
            args.hidden_size, args.intermediate_size,
            rank=args.lora_rank, alpha=args.lora_alpha,
            group_size=args.group_size, bits=args.bits
        )
        self.down_proj = LoRAClass(
            args.intermediate_size, args.hidden_size,
            rank=args.lora_rank, alpha=args.lora_alpha,
            group_size=args.group_size, bits=args.bits
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# Transformer Block with Structured LoRA
# =============================================================================

class StructuredTransformerBlock(nn.Module):
    """Transformer block with structured LoRA."""

    def __init__(self, args: LoRAModelArgs, use_structured: bool = True):
        super().__init__()
        self.self_attn = StructuredAttention(args, use_structured)
        self.mlp = StructuredMLP(args, use_structured)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


# =============================================================================
# Full Model with Structured LoRA
# =============================================================================

class StructuredLoRAModel(nn.Module):
    """Full transformer model with structured LoRA."""

    def __init__(self, args: LoRAModelArgs, use_structured: bool = True):
        super().__init__()
        self.args = args
        self.use_structured = use_structured

        # Embedding (frozen)
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_tokens.freeze()

        # Transformer layers
        self.layers = [
            StructuredTransformerBlock(args, use_structured)
            for _ in range(args.num_hidden_layers)
        ]

        # Final norm (frozen)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm.freeze()

    def __call__(
        self,
        input_ids: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(input_ids)

        # Create causal mask if not provided
        if mask is None:
            L = input_ids.shape[1]
            mask = mx.triu(mx.full((L, L), float('-inf')), k=1)

        for layer in self.layers:
            h = layer(h, mask)

        return self.norm(h)

    def loss(self, input_ids: mx.array, labels: mx.array) -> mx.array:
        """Compute cross-entropy loss."""
        h = self(input_ids)

        # Use embedding weights for output projection (tied embeddings)
        logits = h @ self.embed_tokens.weight.T

        # Cross-entropy loss
        logits = logits[:, :-1, :]  # Shift
        labels = labels[:, 1:]       # Shift

        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction='mean'
        )

        return loss


# =============================================================================
# Helper Functions
# =============================================================================

def create_model(
    args: LoRAModelArgs,
    use_structured: bool = True,
) -> StructuredLoRAModel:
    """Create model with structured or standard LoRA."""
    return StructuredLoRAModel(args, use_structured)


def get_trainable_params(model: nn.Module) -> Dict[str, mx.array]:
    """Get trainable parameters (LoRA A and B matrices)."""
    params = {}
    for name, param in tree_flatten(model.trainable_parameters()):
        params[name] = param
    return params


def count_params(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    return total, trainable
