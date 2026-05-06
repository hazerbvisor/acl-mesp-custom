#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from __future__ import annotations

from typing import Callable
from dataclasses import dataclass, fields, field

import os
import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import load_config, get_model_path
from mlx_lm.models.qwen3 import ModelArgs, Attention, MLP, TransformerBlock, Qwen3Model
from mlx_lm.models.rope_utils import initialize_rope

from .utils import save_params, get_grad_name, product_sum, PAD_VALUE, NULL_LABEL_VALUE
from .modules import (
    LazyLoadingMixin,
    LazyLoadingRMSNorm as RMSNorm,
    LazyLoadingQuantizedLinear as QuantizedLinear,
    LazyLoadingQuantizedEmbedding as QuantizedEmbedding,
)
from .runner import RunFunctionConfig
from .lora import LoRAArgs, lazy_loading_linear_to_lora


@dataclass
class QLoRAModelArgs(ModelArgs):
    bits: int = 4
    group_size: int = 64
    lora_args: LoRAArgs = field(default_factory=lambda: LoRAArgs())

    @classmethod
    def load(cls, path_or_hf_repo: str, **kwargs) -> "QLoRAModelArgs":
        model_path, _ = get_model_path(path_or_hf_repo)
        config = load_config(model_path)
        config_kwargs = {}
        for f in fields(cls):
            if f.name in config:
                config_kwargs[f.name] = config[f.name]
            if f.name in config.get("quantization_config", {}):
                config_kwargs[f.name] = config["quantization_config"][f.name]
        return QLoRAModelArgs(**config_kwargs, **kwargs)


class LazyLoadingAttention(Attention, LazyLoadingMixin):
    def __init__(self, args: QLoRAModelArgs):
        nn.Module.__init__(self)

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        common_kwargs = {"bits": args.bits, "group_size": args.group_size, "bias": False}
        self.q_proj = QuantizedLinear(input_dims=dim, output_dims=n_heads * head_dim, **common_kwargs)
        self.k_proj = QuantizedLinear(input_dims=dim, output_dims=n_kv_heads * head_dim, **common_kwargs)
        self.v_proj = QuantizedLinear(input_dims=dim, output_dims=n_kv_heads * head_dim, **common_kwargs)
        self.o_proj = QuantizedLinear(input_dims=n_heads * head_dim, output_dims=dim, **common_kwargs)

        self.q_norm = RMSNorm(dims=head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(dims=head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def lazy_load(self, params: dict):
        for name in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]:
            getattr(self, name).lazy_load(params[name])


class LazyLoadingMLP(MLP, LazyLoadingMixin):
    def __init__(self, args: QLoRAModelArgs):
        nn.Module.__init__(self)
        common_kwargs = {"bits": args.bits, "group_size": args.group_size, "bias": False}
        dim, hidden_dim = args.hidden_size, args.intermediate_size
        self.gate_proj = QuantizedLinear(input_dims=dim, output_dims=hidden_dim, **common_kwargs)
        self.down_proj = QuantizedLinear(input_dims=hidden_dim, output_dims=dim, **common_kwargs)
        self.up_proj = QuantizedLinear(input_dims=dim, output_dims=hidden_dim, **common_kwargs)

    def lazy_load(self, params: dict):
        for name in ["gate_proj", "down_proj", "up_proj"]:
            getattr(self, name).lazy_load(params[name])


class LazyLoadingTransformerBlock(TransformerBlock, LazyLoadingMixin):
    def __init__(self, args: QLoRAModelArgs):
        nn.Module.__init__(self)
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = LazyLoadingAttention(args)
        self.mlp = LazyLoadingMLP(args)
        self.input_layernorm = RMSNorm(dims=args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            dims=args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def lazy_load(self, params: dict):
        for name in ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"]:
            getattr(self, name).lazy_load(params[name])

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x, mask="causal")


class LazyLoadingLMLoss(nn.Module, LazyLoadingMixin):
    def __init__(self, args: QLoRAModelArgs, reduction: str = 'mean', return_accuracy: bool = False):
        super().__init__()
        self.norm = RMSNorm(dims=args.hidden_size, eps=args.rms_norm_eps)
        self.linear = QuantizedLinear(
            input_dims=args.hidden_size,
            output_dims=args.vocab_size,
            group_size=args.group_size,
            bits=args.bits
        )
        self.reduction = reduction
        self.return_accuracy = return_accuracy

    def lazy_load(self, params: dict):
        self.norm.lazy_load(params["norm"])
        self.linear.lazy_load(params["linear"])

    def __call__(self, x: mx.array, targets: mx.array) -> mx.array | tuple[mx.array, ...]:
        logits = self.linear(self.norm(x))
        mask = mx.logical_and(targets != NULL_LABEL_VALUE, targets != PAD_VALUE).astype(logits.dtype)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")

        if self.reduction == 'mean':
            loss = (losses * mask).sum() / mask.sum()
        elif self.reduction == 'sum':
            loss = (losses * mask).sum()
        else:
            raise ValueError(f"Unsupported reduction method {self.reduction}")

        if self.return_accuracy:
            # Compute token-level accuracy (excluding padding/null labels)
            predictions = mx.argmax(logits, axis=-1)
            correct = (predictions == targets).astype(logits.dtype)
            accuracy = (correct * mask).sum() / mask.sum()
            return loss, accuracy

        return loss


def export_embedding(
    args: QLoRAModelArgs,
    module: nn.QuantizedEmbedding,
    variable_names: dict[str, str],
    output_dir: str,
    save_params_name: str,  # for saving model parameters
    batch_size: int = 1,
    context_length: int = 256,
    shapeless: bool = False,
    module_name: str = "embedding",
    param_name_formatter: Callable[[str], str] = lambda x: x,
    dtype: mx.Dtype = None,  # Target dtype for embedding output
) -> RunFunctionConfig:
    assert "x" in variable_names and "y" in variable_names and isinstance(module, nn.QuantizedEmbedding)
    flatten_param_names, flatten_params = zip(*tree_flatten(module.parameters()))

    # Get target dtype from module if not provided
    if dtype is None:
        dtype = module.scales.dtype

    lazy_module = QuantizedEmbedding(
        num_embeddings=args.vocab_size,
        dims=args.hidden_size,
        bits=args.bits,
        group_size=args.group_size)

    # Capture target_dtype as a concrete value for the closure
    target_dtype = dtype
    # Intermediate dtype to force astype to be traced (not optimized away)
    intermediate_dtype = mx.float32 if dtype != mx.float32 else mx.float16

    def fn(*inputs):
        x, *flatten_params = inputs
        params = tree_unflatten(list(zip(flatten_param_names, flatten_params)))
        lazy_module.lazy_load(params)
        output = lazy_module(x)
        # Force conversion through intermediate dtype to ensure astype is traced
        # Without this, astype(bfloat16) on bfloat16 tensor is a no-op and may be removed
        return output.astype(intermediate_dtype).astype(target_dtype)

    example_x = mx.random.randint(low=0, high=args.vocab_size, shape=(batch_size, context_length))

    fn_path = os.path.join(output_dir, f"{module_name}.mlxfn")
    fn_args = (example_x,) + flatten_params
    mx.export_function(fn_path, fn, *fn_args, shapeless=shapeless)

    params = dict(zip(flatten_param_names, flatten_params))
    params_path = os.path.join(output_dir, save_params_name)
    save_params(params_path, params, param_name_formatter)

    return RunFunctionConfig(
        function_name=module_name,
        input_names=[variable_names["x"]] + [param_name_formatter(n) for n in flatten_param_names],
        output_names=[variable_names["y"]]
    )


def export_transformer_block(
    args: QLoRAModelArgs,
    module: TransformerBlock,
    variable_names: dict[str, str],
    output_dir: str,
    save_params_name: str,  # for saving model parameters
    batch_size: int = 1,
    context_length: int = 256,
    shapeless: bool = False,
    module_name: str = "transformer_block",
    param_name_formatter: Callable[[str], str] = lambda x: x,
    forward_only: bool = False,
) -> tuple[RunFunctionConfig, RunFunctionConfig | None]:
    assert "x" in variable_names and "y" in variable_names and isinstance(module, TransformerBlock)
    trainable_param_names, trainable_params = zip(*tree_flatten(module.trainable_parameters()))
    all_param_names, all_params = zip(*tree_flatten(module.parameters()))

    frozen_param_names, frozen_params = [], []
    for name, param in zip(all_param_names, all_params):
        if name not in trainable_param_names:
            frozen_param_names.append(name)
            frozen_params.append(param)

    # In the flattened params, always put trainable first
    flatten_param_names = trainable_param_names + tuple(frozen_param_names)
    flatten_params = trainable_params + tuple(frozen_params)
    dtype = trainable_params[0].dtype

    argnames = ("x",) + trainable_param_names
    lazy_module = LazyLoadingTransformerBlock(args)
    lazy_loading_linear_to_lora(lazy_module, args.lora_args)

    def forward_fn(*inputs):
        x, *flatten_params = inputs
        params = tree_unflatten(list(zip(flatten_param_names, flatten_params)))
        lazy_module.lazy_load(params)
        # Ensure output dtype matches input dtype (avoids int4 dequant → float32 issue)
        return lazy_module(x).astype(dtype)

    def loss_fn(**kwargs):
        x, dy = kwargs.pop("x"), kwargs.pop("dy")
        params = tree_unflatten(list(zip(kwargs.keys(), kwargs.values())))
        lazy_module.lazy_load(params)
        # Ensure output dtype matches input dtype before computing loss
        return product_sum(lazy_module(x).astype(dtype), dy)

    shape = (batch_size, context_length, args.hidden_size)
    example_x = mx.random.normal(shape=shape, dtype=dtype)

    forward_fn_path = os.path.join(output_dir, f"{module_name}_forward.mlxfn")
    f_args = (example_x,) + flatten_params
    mx.export_function(forward_fn_path, forward_fn, *f_args, shapeless=shapeless)

    trainable_params = dict(zip(trainable_param_names, trainable_params))
    trainable_params_path = os.path.join(output_dir, f"trainable_{save_params_name}")
    save_params(trainable_params_path, trainable_params, param_name_formatter)

    frozen_params = dict(zip(frozen_param_names, frozen_params))
    frozen_params_path = os.path.join(output_dir, save_params_name)
    save_params(frozen_params_path, frozen_params, param_name_formatter)

    x_name, y_name = variable_names["x"], variable_names["y"]
    formatted_param_names = [param_name_formatter(n) for n in flatten_param_names]

    forward_run_config = RunFunctionConfig(
        function_name=f"{module_name}_forward",
        input_names= [x_name] + formatted_param_names,
        output_names=[y_name]
    )

    if forward_only:
        return forward_run_config, None

    def backward_fn(*inputs):
        x, dy, *flatten_params = inputs
        kwargs = dict(zip(flatten_param_names, flatten_params))
        _, grads = mx.grad(loss_fn, argnames=argnames)(**kwargs, x=x, dy=dy)
        # Ensure gradient dtypes match input dtype
        return tuple(grads[name].astype(dtype) for name in argnames)

    example_dy = mx.random.normal(shape=shape, dtype=dtype)
    backward_fn_path = os.path.join(output_dir, f"{module_name}_backward.mlxfn")
    b_args = (example_x, example_dy) + flatten_params
    mx.export_function(backward_fn_path, backward_fn, *b_args, shapeless=shapeless)

    backward_run_config = RunFunctionConfig(
        function_name=f"{module_name}_backward",
        input_names= [x_name, get_grad_name(y_name)] + formatted_param_names,
        output_names=[get_grad_name(x_name)] + [
            get_grad_name(param_name_formatter(n)) for n in trainable_param_names]
    )
    return forward_run_config, backward_run_config


def export_loss(
    args: QLoRAModelArgs,
    module: Qwen3Model,
    variable_names: dict[str, str],
    output_dir: str,
    save_params_name: str,  # for saving model parameters
    batch_size: int = 1,
    context_length: int = 256,
    shapeless: bool = False,
    module_name: str = "loss",
    forward_only: bool = False,
):
    assert "x" in variable_names and "targets" in variable_names and isinstance(module, Qwen3Model)
    linear_param_names, linear_params = zip(*tree_flatten(module.embed_tokens.parameters()))
    norm_param_names, norm_params = zip(*tree_flatten(module.norm.parameters()))
    # Always compute accuracy alongside loss
    lazy_module = LazyLoadingLMLoss(args, return_accuracy=True)

    def loss_fn(*inputs):
        x, targets, *flatten_params = inputs
        linear_params = flatten_params[:len(linear_param_names)]
        norm_params = flatten_params[len(linear_param_names):]
        lazy_module.lazy_load({
            "linear": tree_unflatten(list(zip(linear_param_names, linear_params))),
            "norm": tree_unflatten(list(zip(norm_param_names, norm_params))),
        })
        loss, accuracy = lazy_module(x, targets)
        return loss, accuracy

    def grad_fn(*inputs):
        def loss_only_fn(*inputs):
            loss, _ = loss_fn(*inputs)
            return loss
        loss, x_grad = mx.value_and_grad(loss_only_fn)(*inputs)
        _, accuracy = loss_fn(*inputs)
        return loss, accuracy, x_grad

    dtype = module.norm.weight.dtype
    example_x = mx.random.normal(shape=(batch_size, context_length, args.hidden_size), dtype=dtype)
    example_targets = mx.random.randint(low=0, high=args.vocab_size, shape=(batch_size, context_length))

    fn_path = os.path.join(output_dir, f"{module_name}.mlxfn")
    fn_args = (example_x, example_targets) + linear_params + norm_params

    if forward_only:
        mx.export_function(fn_path, loss_fn, *fn_args, shapeless=shapeless)
        output_names = ["loss", "accuracy"]
    else:
        mx.export_function(fn_path, grad_fn, *fn_args, shapeless=shapeless)
        output_names = ["loss", "accuracy", get_grad_name(variable_names["x"])]

    params = dict(zip(norm_param_names, norm_params))
    params_path = os.path.join(output_dir, save_params_name)
    save_params(params_path, params, lambda x: f"norm.{x}")

    flatten_param_names = list(linear_param_names) + [f"norm.{n}" for n in norm_param_names]
    return RunFunctionConfig(
        function_name=module_name,
        input_names=[variable_names["x"], variable_names["targets"]] + flatten_param_names,
        output_names=output_names
    )
