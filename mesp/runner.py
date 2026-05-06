#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from collections import defaultdict
from typing import Callable
from abc import ABC, abstractmethod

import mlx.core as mx
from mesp.utils import get_grad_name


@dataclass
class RunFunctionConfig:
    function_name: str
    input_names: list[str]
    output_names: list[str]
    checkpoints_to_remove: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def infer_checkpoints_lifetime(configs: list[RunFunctionConfig]):
    output_lifetime = {}

    all_output_names = set()
    for i, config in enumerate(configs):
        for input_name in config.input_names:
            # If a function input depends on the output from previous function
            if input_name in all_output_names:
                # The output has to be alive for this run config
                output_lifetime[input_name] = max(i, output_lifetime.get(input_name, 0))
        all_output_names.update(config.output_names)

    lifetime_to_output_names = defaultdict(list)
    for output_name, lifetime in output_lifetime.items():
        lifetime_to_output_names[lifetime].append(output_name)

    for i, config in enumerate(configs):
        if i in lifetime_to_output_names:
            config.checkpoints_to_remove.extend(lifetime_to_output_names[i])


class FunctionRunner:
    def __init__(self, function_path: str):
        self.function = mx.import_function(function_path)

    def run(self, inputs: dict[str, mx.array], params: dict[str, mx.array], config: RunFunctionConfig) -> dict[str, mx.array]:
        args = []
        for input_name in config.input_names:
            if input_name in inputs:
                args.append(inputs[input_name])
            elif input_name in params:
                args.append(params[input_name])
            else:
                raise ValueError(f"Missing input {input_name} for function {config.function_name} in {params.keys()}")
        outputs = self.function(*args)
        assert len(outputs) == len(config.output_names)
        return dict(zip(config.output_names, outputs))


@dataclass
class GradientsContext(ABC):
    pass


class BaseRunner(ABC):
    def __init__(
        self,
        function_paths: list[str],
        trainable_params_loader: Callable[[], dict[str, mx.array]],
        params_loaders: list[Callable[[], dict[str, mx.array]]],
        configs: list[RunFunctionConfig],
    ):
        self.trainable_params_loader = trainable_params_loader
        self.trainable_params = trainable_params_loader()
        self.function_runners: dict[str, FunctionRunner] = {}
        self.params_loaders: list[Callable[[], dict[str, mx.array]]] = params_loaders
        self.configs: list[RunFunctionConfig] = configs
        for path, config in zip(function_paths, configs):
            if config.function_name not in self.function_runners:
                self.function_runners[config.function_name] = FunctionRunner(path)
        self.checkpoints = {}

    def _run(self, inputs: dict[str, mx.array]):
        self.checkpoints = inputs
        for i, config in enumerate(self.configs):
            params_loader = self.params_loaders[i]
            function_runner = self.function_runners[config.function_name]
            params = params_loader() if params_loader else {}
            outputs = function_runner.run(self.checkpoints, {
                **params, **self.trainable_params
            }, config)
            self.checkpoints.update(outputs)
            for checkpoint in config.checkpoints_to_remove:
                del self.checkpoints[checkpoint]

    def get_metrics(self, metric_names: list[str]) -> dict[str, mx.array]:
        metrics = {}
        for metric_name in metric_names:
            if metric_name in self.checkpoints:
                metrics[metric_name] = self.checkpoints[metric_name]
        return metrics

    @staticmethod
    def accumulate(
        accumulated_grads: dict[str, mx.array] | None,
        grads: dict[str, mx.array]
    ) -> dict[str, mx.array]:
        if accumulated_grads is None:
            return grads

        for name in accumulated_grads:
            assert name in grads
            accumulated_grads[name] += grads[name]

        return accumulated_grads

    @abstractmethod
    def gradients(
        self, inputs: dict[str, mx.array], metric_names: list[str], context: GradientsContext
    ) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
        """ Get gradients of the trainable parameters. """

    def sgd_step(self, accumulated_grads: dict[str, mx.array], learning_rate: float):
        learning_rate = mx.array(learning_rate)
        for name, grad in accumulated_grads.items():
            assert name in self.trainable_params
            self.trainable_params[name] -= learning_rate.astype(grad.dtype) * grad

    def init_adamw_state(self):
        """Initialize AdamW optimizer state (first and second moment estimates)."""
        self.adamw_m = {}  # First moment (mean of gradients)
        self.adamw_v = {}  # Second moment (variance of gradients)
        self.adamw_step_count = 0
        for name, param in self.trainable_params.items():
            self.adamw_m[name] = mx.zeros_like(param)
            self.adamw_v[name] = mx.zeros_like(param)

    def adamw_step(
        self,
        accumulated_grads: dict[str, mx.array],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        AdamW optimizer step.

        Args:
            accumulated_grads: Gradients for trainable parameters
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment (default: 0.9)
            beta2: Exponential decay rate for second moment (default: 0.999)
            eps: Small constant for numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 0.01)
        """
        # Initialize state if needed
        if not hasattr(self, 'adamw_m'):
            self.init_adamw_state()

        self.adamw_step_count += 1
        t = self.adamw_step_count

        # Bias correction terms
        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t

        for name, grad in accumulated_grads.items():
            assert name in self.trainable_params
            param = self.trainable_params[name]

            # Update biased first moment estimate
            self.adamw_m[name] = beta1 * self.adamw_m[name] + (1 - beta1) * grad
            # Update biased second moment estimate
            self.adamw_v[name] = beta2 * self.adamw_v[name] + (1 - beta2) * (grad * grad)

            # Bias-corrected estimates
            m_hat = self.adamw_m[name] / bc1
            v_hat = self.adamw_v[name] / bc2

            # AdamW update: decouple weight decay from gradient update
            # weight decay applied to parameter directly
            update = m_hat / (mx.sqrt(v_hat) + eps)
            self.trainable_params[name] = (
                param * (1 - learning_rate * weight_decay) - learning_rate * update
            ).astype(param.dtype)

    def run(
        self,
        batched_inputs: list[dict[str, mx.array]],
        metric_names: list[str],
        learning_rate: float,
        num_steps: int,
        num_accumulation_steps: int,
        gradients_context: GradientsContext,
    ) -> list[dict[str, mx.array]]:
        accumulated_grads = None
        metrics_list = []
        for i in range(num_steps):
            input_idx = i % len(batched_inputs)
            grads, metrics = self.gradients(batched_inputs[input_idx], metric_names, gradients_context)
            metrics_list.append(metrics)
            # accumulate gradients
            accumulated_grads = self.accumulate(accumulated_grads, grads)
            if (i + 1) % num_accumulation_steps == 0:
                # optimize
                self.sgd_step(accumulated_grads, learning_rate)
                # reset accumulated gradients
                accumulated_grads = None

            print(f"Step: {i+1}, metrics: {metrics}")

        return metrics_list


@dataclass
class MeBPContext(GradientsContext):
    pass


class MeBPRunner(BaseRunner):
    def gradients(
        self, inputs: dict[str, mx.array], metric_names: list[str], context: MeBPContext
    ) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
        self._run(inputs)
        metrics = self.get_metrics(metric_names)
        grads = {
            name: self.checkpoints[get_grad_name(name)]
            for name in self.trainable_params
        }
        self.checkpoints = {}
        return grads, metrics

