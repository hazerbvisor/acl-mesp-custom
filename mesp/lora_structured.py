#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
LoRA-Structured Backprop Implementation for MLX.

This module implements memory-efficient backward propagation for LoRA layers
by using explicit gradient formulas instead of generic autograd.

Key benefits:
1. Reduced activation memory (store only x, recompute h)
2. Smaller intermediate tensor sizes (low-rank matrices)
3. Same mathematical gradients as standard LoRA

Based on the equations from idea.md:
- Forward: h = Ax, Δy = Bh, y = W₀x + s·Δy
- Backward: grad_B = (s·g)hᵀ, grad_A = (Bᵀ(s·g))xᵀ, grad_x = W₀ᵀg + Aᵀ(Bᵀ(s·g))
"""

import mlx.core as mx
import mlx.nn as nn


@mx.custom_function
def lora_linear_forward(
    x: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
    scale: mx.array,
) -> mx.array:
    """
    LoRA forward pass (without base linear).

    Args:
        x: Input tensor (batch, seq_len, in_features)
        lora_a: LoRA A matrix (in_features, rank)
        lora_b: LoRA B matrix (rank, out_features)
        scale: Scaling factor (scalar)

    Returns:
        delta_y: LoRA output contribution (batch, seq_len, out_features)
    """
    # h = x @ A  (batch, seq_len, rank)
    h = x @ lora_a
    # delta_y = h @ B  (batch, seq_len, out_features)
    delta_y = h @ lora_b
    # Apply scaling
    return scale * delta_y


@lora_linear_forward.vjp
def lora_linear_backward(
    primals: tuple,
    cotangent: mx.array,
    outputs: tuple,
) -> tuple:
    """
    Custom backward for LoRA layer using structured gradient computation.

    This implements Option A from idea.md: store only x, recompute h.

    Args:
        primals: (x, lora_a, lora_b, scale) - saved from forward
        cotangent: grad_delta_y - gradient from downstream (single array)
        outputs: (delta_y,) - forward outputs (not used)

    Returns:
        Gradients for (x, lora_a, lora_b, scale)
    """
    x, lora_a, lora_b, scale = primals
    grad_output = cotangent

    # Scaled gradient
    # grad_output: (batch, seq_len, out_features)
    grad_scaled = scale * grad_output

    # Recompute h (Option A - memory efficient)
    # h: (batch, seq_len, rank)
    h = x @ lora_a

    # Gradient for B: grad_B = h^T @ grad_scaled
    # We need to sum over batch and seq dimensions
    # h: (batch, seq_len, rank), grad_scaled: (batch, seq_len, out_features)
    # grad_b should be (rank, out_features)
    grad_b = mx.einsum('bsr,bso->ro', h, grad_scaled)

    # Gradient for h: grad_h = grad_scaled @ B^T
    # grad_h: (batch, seq_len, rank)
    grad_h = grad_scaled @ lora_b.T

    # Gradient for A: grad_A = x^T @ grad_h
    # x: (batch, seq_len, in_features), grad_h: (batch, seq_len, rank)
    # grad_a should be (in_features, rank)
    grad_a = mx.einsum('bsi,bsr->ir', x, grad_h)

    # Gradient for x: grad_x = grad_h @ A^T
    # grad_x: (batch, seq_len, in_features)
    grad_x = grad_h @ lora_a.T

    # Gradient for scale (not typically needed, but included for completeness)
    # grad_scale = sum(grad_output * delta_y_unscaled)
    delta_y_unscaled = h @ lora_b
    grad_scale = mx.sum(grad_output * delta_y_unscaled)

    return grad_x, grad_a, grad_b, grad_scale


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
    Complete QLoRA forward pass (quantized base + LoRA).

    Args:
        x: Input tensor (batch, seq_len, in_features)
        w0_weight: Quantized base weight
        w0_scales: Quantization scales
        w0_biases: Quantization biases
        lora_a: LoRA A matrix (in_features, rank)
        lora_b: LoRA B matrix (rank, out_features)
        scale: LoRA scaling factor
        group_size: Quantization group size
        bits: Quantization bits

    Returns:
        y: Output tensor (batch, seq_len, out_features)
    """
    # Base linear with quantized weights
    # Use MLX's quantized matmul
    y_base = mx.quantized_matmul(
        x, w0_weight, w0_scales, w0_biases,
        transpose=True,
        group_size=int(group_size.item()),
        bits=int(bits.item())
    )

    # LoRA contribution
    h = x @ lora_a
    delta_y = h @ lora_b
    y_lora = scale * delta_y

    return y_base + y_lora


@lora_qlinear_forward.vjp
def lora_qlinear_backward(
    primals: tuple,
    cotangent: mx.array,
    outputs: tuple,
) -> tuple:
    """
    Custom backward for QLoRA layer.

    Only computes gradients for trainable LoRA parameters (lora_a, lora_b).
    Base weights are frozen, so their gradients are not computed.

    Args:
        primals: Forward inputs
        cotangent: Gradient from downstream (single array)
        outputs: Forward outputs (not used)

    Returns:
        Gradients for all inputs (None for frozen weights)
    """
    x, w0_weight, w0_scales, w0_biases, lora_a, lora_b, scale, group_size, bits = primals
    grad_output = cotangent

    # Scaled gradient for LoRA branch
    grad_scaled = scale * grad_output

    # Recompute h (Option A - memory efficient)
    h = x @ lora_a

    # Gradient for B: grad_B = h^T @ grad_scaled
    grad_b = mx.einsum('bsr,bso->ro', h, grad_scaled)

    # Gradient for h: grad_h = grad_scaled @ B^T
    grad_h = grad_scaled @ lora_b.T

    # Gradient for A: grad_A = x^T @ grad_h
    grad_a = mx.einsum('bsi,bsr->ir', x, grad_h)

    # Gradient for x (from both branches)
    # From LoRA: grad_x_lora = grad_h @ A^T
    grad_x_lora = grad_h @ lora_a.T

    # From base: grad_x_base = grad_output @ W0
    # For quantized weights, we need to dequantize for backward
    grad_x_base = mx.quantized_matmul(
        grad_output, w0_weight, w0_scales, w0_biases,
        transpose=False,
        group_size=int(group_size.item()),
        bits=int(bits.item())
    )

    grad_x = grad_x_lora + grad_x_base

    # Gradient for scale
    delta_y_unscaled = h @ lora_b
    grad_scale = mx.sum(grad_output * delta_y_unscaled)

    # Return None for frozen weights (w0_weight, w0_scales, w0_biases, group_size, bits)
    # MLX custom_function expects gradients for all primals
    # For frozen parameters, we return zero gradients
    return (
        grad_x,
        mx.zeros_like(w0_weight),  # w0_weight (frozen)
        mx.zeros_like(w0_scales),  # w0_scales (frozen)
        mx.zeros_like(w0_biases),  # w0_biases (frozen)
        grad_a,
        grad_b,
        grad_scale,
        mx.array(0.0),  # group_size (not differentiable)
        mx.array(0.0),  # bits (not differentiable)
    )


class StructuredLoRALinear(nn.Module):
    """
    LoRA Linear layer with structured backward propagation.

    This layer wraps a frozen base linear layer with trainable LoRA matrices,
    using custom backward kernels for memory-efficient gradient computation.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        rank: int = 8,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.rank = rank
        self._scale = scale

        # Base linear (frozen)
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.linear.freeze()

        # LoRA matrices (trainable)
        # A: (in_features, rank)
        # B: (rank, out_features)
        lora_scale = 1.0 / rank
        self.lora_a = mx.random.normal(shape=(input_dims, rank)) * lora_scale
        self.lora_b = mx.zeros((rank, output_dims))

    def __call__(self, x: mx.array) -> mx.array:
        # Base forward
        y_base = self.linear(x)

        # LoRA forward with structured backward
        scale = mx.array(self._scale)
        delta_y = lora_linear_forward(x, self.lora_a, self.lora_b, scale)

        return y_base + delta_y


class StructuredQLoRALinear(nn.Module):
    """
    Quantized LoRA Linear layer with structured backward propagation.

    This is the memory-efficient version that uses:
    1. Quantized base weights (4-bit)
    2. Custom backward that recomputes h instead of storing it
    """

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
        self._scale = scale
        self.group_size = group_size
        self.bits = bits

        # Base quantized linear (frozen)
        self.linear = nn.QuantizedLinear(
            input_dims, output_dims, bias=bias,
            group_size=group_size, bits=bits
        )
        self.linear.freeze()

        # LoRA matrices (trainable)
        lora_scale = 1.0 / rank
        self.lora_a = mx.random.normal(shape=(input_dims, rank)) * lora_scale
        self.lora_b = mx.zeros((rank, output_dims))

    def __call__(self, x: mx.array) -> mx.array:
        # Use the combined forward with structured backward
        scale = mx.array(self._scale)
        group_size = mx.array(self.group_size)
        bits = mx.array(self.bits)

        return lora_qlinear_forward(
            x,
            self.linear.weight,
            self.linear.scales,
            self.linear.biases,
            self.lora_a,
            self.lora_b,
            scale,
            group_size,
            bits,
        )

    @staticmethod
    def from_base(
        linear: nn.QuantizedLinear,
        rank: int = 8,
        scale: float = 20.0,
    ) -> "StructuredQLoRALinear":
        """Create from an existing quantized linear layer."""
        layer = StructuredQLoRALinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            rank=rank,
            scale=scale,
            bias=linear.bias is not None,
            group_size=linear.group_size,
            bits=linear.bits,
        )
        # Copy the base weights
        layer.linear.weight = linear.weight
        layer.linear.scales = linear.scales
        layer.linear.biases = linear.biases
        if linear.bias is not None:
            layer.linear.bias = linear.bias
        layer.linear.freeze()
        return layer
