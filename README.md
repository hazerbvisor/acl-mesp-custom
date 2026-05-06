# MeSP: Memory-Efficient Structured Backpropagation for On-Device LLM Fine-Tuning

Official implementation of **MeSP: Memory-Efficient Structured Backpropagation
for On-Device LLM Fine-Tuning**, accepted to the ACL 2026 Industry Track.

MeSP exploits LoRA's low-rank structure to derive backward passes manually,
recomputing the small intermediate projection `h = xA` during backward
(`r << d_in`, so this is cheap) instead of storing it. By processing
transformer blocks sequentially and explicitly controlling tensor lifetimes
that automatic differentiation cannot, MeSP **achieves 49% average peak-memory
reduction over MeBP on Qwen2.5 (0.5B–3B) while computing mathematically
identical gradients** — reducing peak memory from 361 MB to 136 MB on
Qwen2.5-0.5B at sequence length 256.

## Repository structure

```
.
├── mesp/             Python (MLX) package — base models, LoRA, and the MeSP
│                     forward / backward modules for Qwen2 / Qwen3 / Gemma3
├── MLXMeSP/          Swift on-device runtime library (LoRAStructuredRunner.swift)
├── MLXMeSPTests/     Swift unit tests (`swift test`)
├── MLXMeSPExample/   Reference iOS / macOS demo app (Xcode project)
├── export/           Output directory for compiled `.mlxfn` model artifacts
│                     consumed by the iOS app
├── Package.swift     Swift package manifest
├── Package.resolved  Locked Swift dependencies
├── requirements.txt  Python dependencies
├── LICENSE
└── README.md
```

The implementation extends Apple's
[MeBP](https://github.com/apple/ml-mebp) runtime; baseline MeBP code paths are
retained for direct comparison. The `*_structured*.py` Python modules and
`MLXMeSP/LoRAStructuredRunner.swift` contain the MeSP contribution.

## Requirements

The MLX backend requires Apple Silicon (M1/M2/M3/M4 or Apple A-series). Tested
on macOS 14+ with Python 3.10+ and Xcode 15+.

Python dependencies:

```setup
pip install -r requirements.txt
```

Swift dependencies are resolved automatically by SwiftPM / Xcode from
`Package.swift`.

## Quick start: on-device demo

The fastest way to see MeSP working end-to-end is the bundled iOS / macOS app:

```bash
open MLXMeSPExample/MLXMeSPExample.xcodeproj
```

Then build and run on a device. The app links against the `MLXMeSP` Swift
library, ships with `wikitext2.jsonl` training data, loads pre-compiled
`.mlxfn` model artifacts from `export/`, and runs LoRA fine-tuning live with
on-screen loss / memory metrics.

To use a different base model, run the export pipeline (Python) to populate
`export/<model-name>/` with the required `embedding.mlxfn`,
`layer{i}_forward_swift.mlxfn`, `loss.mlxfn`, and `run_configs.json` files,
then point the app at that directory.

## Python API

The `mesp/` package exposes the MeSP primitives so they can be embedded into
your own training loop:

```python
import mlx.core as mx
from mesp.lora import LoRAArgs, load_qlora_model
from mesp.lora_structured import (
    lora_qlinear_forward,        # @mx.custom_function with structured VJP
    StructuredQLoRALinear,       # drop-in nn.Module replacement
)

# Load a 4-bit-quantised LLM and convert q/k/v/o/gate/up/down projections
# to LoRA layers with rank 8.
args = LoRAArgs(rank=8, scale=20.0)
model, tokenizer = load_qlora_model(
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit", args,
)

# Standard MLX value_and_grad now uses the structured backward
# automatically — no other code changes required.
loss_and_grad = mx.value_and_grad(loss_fn)
loss, grads = loss_and_grad(params, input_ids, labels)
```

Model-specific transformer-block exports (which the Swift runner consumes)
are in `mesp/qwen2_structured.py`, `mesp/qwen3_structured.py`,
`mesp/gemma3_structured.py`, and the `*_structured_swift_optimized.py`
variants that emit the 8 intermediate tensors required for Swift-side
backward.

## Swift testing

Build the library and run unit tests:

```bash
swift build -c release
swift test
```

## On-device implementation details

`MLXMeSP/LoRAStructuredRunner.swift` implements the structured backward in
Swift via the primitives `loraLinearBackward`, `rmsNormBackward`,
`softmaxBackward`, `siluBackward`, and `gqaBackward`. Each transformer-block
forward function exported from Python returns the layer output plus
**8 intermediate tensors** required for backward (Section 4.5 of the paper).
After every per-layer backward, `GPU.clearCache()` is invoked to ensure peak
memory matches a single layer's working set rather than the cumulative
working set across layers.

## Pre-trained models

MeSP fine-tunes existing 4-bit-quantized checkpoints with LoRA adapters; we
do not release new base weights. The base checkpoints used in the paper are
available on the Hugging Face Hub:

- [`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit)
- [`mlx-community/Qwen2.5-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit)
- [`mlx-community/Qwen2.5-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit)

## Results

All numbers below were reported in the paper with batch size 1, LoRA rank 8
(target modules `q, k, v, o, gate, up, down`), 4-bit (group-size 64)
quantized base, bfloat16 LoRA, SGD, learning rate `1e-4`. Memory uses the
`phys_footprint` field of the iOS/macOS `task_info` API. Tables 1–3 are
measured on **iPhone 17 Pro (A19 Pro, 8 GB RAM)**; the convergence study
(Figure 2) uses an Apple Silicon **M4**.

### Table 1 — Memory & time at sequence length 256

| Model | Method | Memory (MB) | Time (s) | Reduction |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | MeBP | 360.8 | 0.68 | — |
| Qwen2.5-0.5B | **MeSP** | **136.2** | 0.86 | **62%** |
| Qwen2.5-1.5B | MeBP | 516.2 | 1.66 | — |
| Qwen2.5-1.5B | **MeSP** | **262.6** | 2.17 | **49%** |
| Qwen2.5-3B   | MeBP | 637.6 | 3.21 | — |
| Qwen2.5-3B   | **MeSP** | **368.4** | 4.09 | **42%** |

### Table 2 — Sequence-length scaling (Qwen2.5-0.5B, peak memory in MB)

| Method | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|
| MeBP | 252.7 | 360.8 | 582.4 | 1050.3 |
| **MeSP** | **110.7** | **136.2** | **245.8** | **513.6** |
| **MeSP reduction** | **56%** | **62%** | **58%** | **51%** |

### Table 3 — LoRA-rank sensitivity (Qwen2.5-0.5B, seq=256, peak memory in MB)

| Method | r=4 | r=8 | r=16 | r=32 |
|---|---:|---:|---:|---:|
| MeBP | 355.2 | 360.8 | 372.4 | 395.8 |
| **MeSP** | **132.8** | **136.2** | **143.5** | **158.2** |
| **MeSP reduction** | **63%** | **62%** | **61%** | **60%** |

### Table 5 — h strategy ablation (Qwen2.5-3B, seq=256)

| Strategy | Memory (MB) | Time (s) |
|---|---:|---:|
| MeBP (baseline) | 637.6 | 3.21 |
| Store h | 398.5 | 3.85 |
| **Recompute h (ours)** | **368.4** | **4.09** |

Recomputing `h` saves an additional 7.6% memory at a 6.2% time cost,
validating the design principle "recompute small tensors instead of storing
them" across all 168 LoRA layers (24 blocks × 7 projections).

### Gradient-correctness check

Running training with identical seeds on MeBP and MeSP yields **bit-exact
loss matches at every step** (deviation < 1e-6), confirming that MeSP
computes mathematically identical gradients to standard backpropagation.

## Citation

```bibtex
@inproceedings{park2026mesp,
  title     = {{MeSP}: Memory-Efficient Structured Backpropagation for
               On-Device {LLM} Fine-Tuning},
  author    = {Park, Junyoung and Hong, Yuri and Kim, Sungwan and Lee, Jaeho},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for
               Computational Linguistics: Industry Track},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  organization = {Opt-AI Inc.}
}
```

## License

This project builds on Apple's [MeBP](https://github.com/apple/ml-mebp)
runtime and inherits its license. See `LICENSE`.

## Contributing

Bug reports and pull requests are welcome. Please open an issue describing
the change before submitting a PR.
