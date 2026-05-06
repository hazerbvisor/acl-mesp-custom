#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from abc import ABC, ABCMeta, abstractmethod

from mlx_lm.tuner.lora import LoRALinear
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gemma3_text import RMSNorm as GemmaRMSNorm




class LazyLoadingMixin(ABC):
    @abstractmethod
    def lazy_load(self, params: dict):
        """ Assign weights to module """


class LazyLoadingMeta(ABCMeta):
    def __new__(cls, name, bases, attrs, skip_classes=None):
        if skip_classes is None:
            skip_classes = []

        original_init = attrs.get('__init__')

        def new_init(self, **kwargs):
            # Find the appropriate parent to call based on MRO
            mro = type(self).__mro__

            # Skip specified classes
            for base_class in mro[1:]:  # Skip self
                if base_class not in skip_classes and hasattr(base_class, '__init__'):
                    assert base_class == nn.Module
                    base_class.__init__(self)
                    break

            # Call the original __init__ if it exists
            if original_init:
                original_init(self, **kwargs)

            for key, value in kwargs.items():
                setattr(self, key, value)

        attrs['__init__'] = new_init
        assert "lazy_load" in attrs, "lazy_load must be implemented in subclass"
        return super().__new__(cls, name, bases, attrs)


class LazyLoadingLinear(
    nn.Linear,
    LazyLoadingMixin,
    metaclass=LazyLoadingMeta,
    skip_classes=[nn.Linear]
):
    def lazy_load(self, params: dict[str, mx.array]):
        weight = params["weight"]
        assert weight.shape == (self.output_dims, self.input_dims)
        self.weight = weight
        if "bias" in params:
            assert params["bias"].shape == (self.output_dims,)
            self.bias = params["bias"]


class LazyLoadingQuantizedLinear(
    nn.QuantizedLinear,
    LazyLoadingMixin,
    metaclass=LazyLoadingMeta,
    skip_classes=[nn.QuantizedLinear]
):
    def lazy_load(self, params: dict[str, mx.array]):
        self.weight = params["weight"]
        self.scales = params["scales"]
        self.biases = params["biases"]
        assert self.weight.dtype == mx.uint32
        input_dims = self.input_dims * self.bits // 32
        assert self.weight.shape == (self.output_dims, input_dims)
        if "bias" in params:
            assert params["bias"].shape == (self.output_dims,)
            self.bias = params["bias"]
        self.freeze()


class LazyLoadingQuantizedEmbedding(
    nn.QuantizedEmbedding, LazyLoadingMixin,
    metaclass=LazyLoadingMeta, skip_classes=[nn.QuantizedEmbedding]
):
    def lazy_load(self, params: dict[str, mx.array]):
        self.weight = params["weight"]
        self.scales = params["scales"]
        self.biases = params["biases"]
        assert self.weight.dtype == mx.uint32
        dims = self.dims * self.bits // 32
        assert self.weight.shape == (self.num_embeddings, dims)
        self.freeze()


class LazyLoadingRMSNorm(
    nn.RMSNorm,
    LazyLoadingMixin,
    metaclass=LazyLoadingMeta,
    skip_classes=[nn.RMSNorm]
):
    def lazy_load(self, params: dict[str, mx.array]):
        self.weight = params["weight"]

class LazyLoadingGemmaRMSNorm(
    GemmaRMSNorm,
    LazyLoadingMixin,
    metaclass=LazyLoadingMeta,
    skip_classes=[GemmaRMSNorm],
):
    def lazy_load(self, params: dict[str, mx.array]):
        self.weight = params["weight"]


class LazyLoadingQLoraLinear(LoRALinear, LazyLoadingMixin):
    # Similar to https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/lora.py
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
    ):
        nn.Module.__init__(self)
        self.linear = LazyLoadingQuantizedLinear(
            input_dims=input_dims,
            output_dims=output_dims,
            bias=bias,
            group_size=group_size,
            bits=bits)
        self.dropout = nn.Dropout(p=dropout)
        self.r = r
        self.scale = scale

    def lazy_load(self, params: dict):
        self.linear.lazy_load(params["linear"])
        assert params["lora_a"].shape == (self.linear.input_dims, self.r)
        assert params["lora_b"].shape == (self.r, self.linear.output_dims)
        self.lora_a = params["lora_a"]
        self.lora_b = params["lora_b"]

    @staticmethod
    def from_base(
        linear: LazyLoadingQuantizedLinear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        # on linear and quantized linear
        lora_lin = LazyLoadingQLoraLinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
            bias = linear.bias
        )
        lora_lin.linear = linear
        return lora_lin


# Lazy import for structured LoRA - only needed at runtime, not during export
# The @mx.custom_function decorator is incompatible with mx.export_function
_lora_linear_forward = None

def _get_lora_linear_forward():
    """Lazy load lora_linear_forward to avoid import issues during export."""
    global _lora_linear_forward
    if _lora_linear_forward is None:
        from mesp.lora_structured import lora_linear_forward
        _lora_linear_forward = lora_linear_forward
    return _lora_linear_forward


class LazyLoadingStructuredQLoraLinear(nn.Module, LazyLoadingMixin):
    """
    QLoRA Linear with structured backward propagation for memory efficiency.

    Uses custom backward kernels that:
    1. Store only x in checkpoints (not h = Ax)
    2. Recompute h during backward (cheap due to low rank)
    3. Compute gradients using explicit formulas
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()
        self.linear = LazyLoadingQuantizedLinear(
            input_dims=input_dims,
            output_dims=output_dims,
            bias=bias,
            group_size=group_size,
            bits=bits)
        self.dropout = nn.Dropout(p=dropout)
        self.r = r
        self._scale = scale
        self.input_dims = input_dims
        self.output_dims = output_dims

        # LoRA matrices will be loaded via lazy_load
        # Shape: lora_a (input_dims, r), lora_b (r, output_dims)
        self.lora_a = None
        self.lora_b = None

    def lazy_load(self, params: dict):
        self.linear.lazy_load(params["linear"])
        assert params["lora_a"].shape == (self.input_dims, self.r)
        assert params["lora_b"].shape == (self.r, self.output_dims)
        self.lora_a = params["lora_a"]
        self.lora_b = params["lora_b"]

    def __call__(self, x: mx.array) -> mx.array:
        # Base linear forward (quantized)
        y_base = self.linear(x)

        # LoRA forward with structured backward
        # Uses custom VJP for memory-efficient gradient computation
        scale = mx.array(self._scale)
        lora_forward = _get_lora_linear_forward()
        delta_y = lora_forward(x, self.lora_a, self.lora_b, scale)

        return y_base + delta_y

    @staticmethod
    def from_base(
        linear: LazyLoadingQuantizedLinear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        """Create from an existing quantized linear layer."""
        lora_lin = LazyLoadingStructuredQLoraLinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
            bias=linear.bias is not None,
            group_size=linear.group_size,
            bits=linear.bits,
        )
        lora_lin.linear = linear
        return lora_lin
