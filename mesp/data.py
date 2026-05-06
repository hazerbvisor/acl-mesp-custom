#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from __future__ import annotations

"""
Data loading utilities for MeBP training experiments.

Provides WikiText-2 dataset loader with configurable sequence length and sample count.
"""

from typing import Iterator, Optional
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class DataConfig:
    """Configuration for dataset loading."""
    sequence_length: int = 256
    num_samples: int = 2048
    batch_size: int = 1
    shuffle: bool = True


def load_wikitext2(
    tokenizer,
    config: DataConfig,
    split: str = "train",
) -> list[dict[str, mx.array]]:
    """
    Load WikiText-2 dataset and prepare batches.

    Args:
        tokenizer: HuggingFace tokenizer
        config: Data configuration
        split: Dataset split ("train", "validation", "test")

    Returns:
        List of batched inputs with "input_ids" and "label_ids"
    """
    from datasets import load_dataset

    # Load WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Concatenate all text
    all_text = "\n".join([text for text in dataset["text"] if text.strip()])

    # Tokenize
    tokens = tokenizer.encode(all_text)
    tokens = mx.array(tokens)

    # Create sequences of length sequence_length + 1 (for input and label)
    seq_len = config.sequence_length
    num_sequences = min(
        len(tokens) // (seq_len + 1),
        config.num_samples
    )

    # Trim to exact number of sequences
    tokens = tokens[:num_sequences * (seq_len + 1)]
    tokens = tokens.reshape(num_sequences, seq_len + 1)

    # Input is first seq_len tokens, label is shifted by 1
    input_ids = tokens[:, :-1]  # (num_sequences, seq_len)
    label_ids = tokens[:, 1:]   # (num_sequences, seq_len)

    # Create batches
    batches = []
    indices = list(range(num_sequences))

    if config.shuffle:
        import random
        random.shuffle(indices)

    for i in range(0, num_sequences, config.batch_size):
        batch_indices = indices[i:i + config.batch_size]
        if len(batch_indices) < config.batch_size:
            continue  # Skip incomplete batches

        batch_input = mx.stack([input_ids[j] for j in batch_indices])
        batch_label = mx.stack([label_ids[j] for j in batch_indices])

        batches.append({
            "input_ids": batch_input,
            "label_ids": batch_label,
        })

    return batches


def create_data_iterator(
    batches: list[dict[str, mx.array]],
    num_steps: int,
) -> Iterator[dict[str, mx.array]]:
    """
    Create an iterator that cycles through batches for num_steps.

    Args:
        batches: List of batch dictionaries
        num_steps: Total number of training steps

    Yields:
        Batch dictionary with input_ids and label_ids
    """
    for i in range(num_steps):
        yield batches[i % len(batches)]


def get_tokenizer(model_path: str):
    """
    Load tokenizer from model path.

    Args:
        model_path: Path to model directory

    Returns:
        HuggingFace tokenizer
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
