#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from __future__ import annotations

from dataclasses import dataclass, field

import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from transformers import PreTrainedTokenizer
from mesp.modules import (
    LazyLoadingQLoraLinear,
    LazyLoadingStructuredQLoraLinear,
    LazyLoadingQuantizedLinear,
    LazyLoadingMixin,
)


@dataclass
class LoRAArgs:
    rank: int = 8
    dropout: float = 0.0
    scale: float = 20.0
    use_structured_backward: bool = True  # Use memory-efficient structured backward
    target_modules: list[str] = field(
        default_factory=lambda: [
            "self_attn.q_proj",
            "self_attn.v_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "mlp.down_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
        ]
    )


def load_qlora_model(
    path_or_hf_repo: str, args: LoRAArgs
) -> tuple[nn.Module, PreTrainedTokenizer]:
    model, tokenizer = load(path_or_hf_repo)
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), config={
        "rank": args.rank,
        "dropout": args.dropout,
        "scale": args.scale,
        "keys": args.target_modules
    })
    return model, tokenizer


def lazy_loading_linear_to_lora(model: nn.Module, args: LoRAArgs):
    """
    Convert linear layers to LoRA layers.

    Args:
        model: The model to convert
        args: LoRA configuration including:
            - rank: LoRA rank
            - dropout: Dropout rate
            - scale: LoRA scaling factor
            - use_structured_backward: If True, use memory-efficient structured backward
            - target_modules: List of module names to convert
    """
    assert isinstance(model, LazyLoadingMixin) and isinstance(model, nn.Module)
    keys = set(args.target_modules)

    # Select LoRA class based on structured_backward option
    lora_class = (
        LazyLoadingStructuredQLoraLinear
        if args.use_structured_backward
        else LazyLoadingQLoraLinear
    )

    def to_lora(module: nn.Module):
        if isinstance(module, LazyLoadingQuantizedLinear):
            return lora_class.from_base(
                module,
                r=args.rank,
                dropout=args.dropout,
                scale=args.scale
            )
        else:
            raise ValueError(f"Unsupported module: {type(module)}")

    if hasattr(model, "layers"):
        num_layers = len(model.layers)
        for l in model.layers[-max(num_layers, 0):]:
            lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if
                           k in keys]
            if lora_layers:
                l.update_modules(tree_unflatten(lora_layers))

    lora_modules = [(k, to_lora(m)) for k, m in model.named_modules() if
                    k in keys]
    if lora_modules:
        model.update_modules(tree_unflatten(lora_modules))
