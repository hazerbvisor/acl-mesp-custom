#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from __future__ import annotations

import shutil
from typing import Callable
import subprocess
import os
import mlx.core as mx
from transformers import PreTrainedTokenizer
from huggingface_hub import hf_hub_download


PAD_VALUE = 0
NULL_LABEL_VALUE = -100


def dict_to_arrays(
        d: dict[str, mx.array], sort: bool = False
) -> tuple[list[str], list[mx.array]]:
    keys = list(d.keys())
    if sort:
        keys.sort()
    return keys, [d[k] for k in keys]


def arrays_to_dict(
        keys: list[str],
        values: list[mx.array] | tuple[mx.array, ...]
) -> dict[str, mx.array]:
    assert len(keys) == len(values)
    return dict(zip(keys, values))


def save_params(
        save_path: str,
        params: dict[str, mx.array],
        key_formatter: Callable[[str], str] = lambda x: x,
):
    mx.save_safetensors(save_path, {key_formatter(key): val for key, val in params.items()})


def get_grad_name(x: str) -> str:
    return f"{x}.grad"


@mx.custom_function
def product_sum(x, y):
    return mx.sum(x * y)


@product_sum.vjp
def product_sum_vjp(primals, cotangents, _):
    x, y = primals
    return cotangents * y, cotangents * x


def process_messages_data(
        data: list[list[dict[str, str]]],
        tokenizer: PreTrainedTokenizer,
        input_ids_key: str = "input_ids",
        label_ids_key: str = "label_ids",
        max_length: int = 256,
        pad_to_max_length: bool = True,
) -> list[dict[str, mx.array]]:
    processed_data = []

    def process(messages) -> dict[str, mx.array]:
        assert (len(messages) == 2 and
                messages[0]["role"] == "user" and
                messages[1]["role"] == "assistant")
        tokens = tokenizer.apply_chat_template(messages)
        offset = len(tokenizer.apply_chat_template([messages[0]]))
        assert offset < max_length

        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        # tokens up to offset is masked        
        labels = [NULL_LABEL_VALUE] * (offset - 1) + tokens[offset:] + [PAD_VALUE]

        if pad_to_max_length and len(tokens) < max_length:
            tokens = tokens + [PAD_VALUE] * (max_length - len(tokens))
            labels = labels + [NULL_LABEL_VALUE] * (max_length - len(labels))

        return {
            input_ids_key: mx.array(tokens).reshape(1, max_length),
            label_ids_key: mx.array(labels).reshape(1, max_length),
        }

    for messages in data:
        processed_data.append(process(messages))

    return processed_data


def prepare_device_tokenizer_assets(
    repo_id: str,
    output_dir: str,
    local_only: bool = False,
    filenames: list[str] = ["config.json", "tokenizer_config.json", "tokenizer.json"],
):
    os.makedirs(output_dir, exist_ok=True)
    for filename in filenames:
        if local_only:
            try:
                cmd = ["cp", os.path.join(repo_id, filename), os.path.join(output_dir, filename)]
                subprocess.run(cmd, check=True)
                print(f"cp {filename} to {output_dir}")
            except Exception as e:
                print(f"Skipped {filename}: {e}")
        else:
            try:
                path = hf_hub_download(repo_id, filename, local_dir=output_dir)
                print(f"Downloaded {filename} to {path}")
            except Exception as e:
                print(f"Skipped {filename}: {e}")
    cache_dir = os.path.join(output_dir, ".cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

