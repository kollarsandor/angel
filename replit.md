# JAIDE v40 - Zig-based LLM System (RSF Architecture)

## Overview
JAIDE is a from-scratch LLM system built in Zig 0.14.1 using a custom RSF (Relational Spectral Flow) architecture with affin coupling. It includes a custom tokenizer (MGT), neural network training pipeline, inference server, and optional GPU/FPGA acceleration.

## Architecture
- **Language**: Zig 0.14.1 (primary), Futhark (GPU kernels), Python (Modal training scripts)
- **Model**: RSF architecture with affin coupling (scale + translation), configurable dimensions and layers
- **Tokenizer**: MGT (Magyar/Hungarian Graph Tokenizer) with ~475 vocab tokens
- **Training**: Local via CLI or remote via Modal (GPU cloud)
- **Dataset**: HuggingFaceFW/finephrase (auto-downloaded in Modal scripts)

## GPU Pipeline (Futhark)
The GPU pipeline in `main.fut` implements the same affin coupling logic as the Zig-side `rsf.zig`:
1. Split each row into x1 (first half) and x2 (second half)
2. Scale: s = exp(clip(W_s * x2 + b_s, clip_min, clip_max))
3. y1 = x1 * s (elementwise)
4. Translation: t = W_t * y1 + b_t
5. y2 = x2 + t (elementwise)
6. Output: y1 ++ y2

The backward pass reconstructs inputs from outputs and computes gradients for weights and biases.

## Key Files
- `build.zig` / `build.zig.zon` - Build system configuration
- `src/main.zig` - Main entry point, training loop, dataset loading
- `src/processor/rsf.zig` - RSF model with affin coupling (CPU + GPU dispatch)
- `src/hw/accel/main.fut` - GPU forward/backward/training (Futhark, affin coupling)
- `src/hw/accel/futhark_bindings.zig` - Zig bindings for Futhark GPU entries
- `src/hw/accel/accel_interface.zig` - RSFAccelerator with bias/clip support
- `src/hw/accel/futhark_kernels.fut` - Standalone GPU kernels (scatter, flow, etc.)
- `src/tokenizer/mgt.zig` - Custom tokenizer with Hungarian vocabulary
- `src/distributed/distributed_trainer_futhark.zig` - Distributed GPU training

## Build Commands
- `zig build` - Default build (CPU)
- `zig build -Dgpu=true -Doptimize=ReleaseFast` - GPU build
- `zig build run` - Run interactive JAIDE console

## Dataset Pipeline
- Both Modal scripts auto-download `HuggingFaceFW/finephrase` dataset
- Converts to JSONL format with `{"text": "..."}` per line
- `loadDatasetSamples` in main.zig extracts text from JSON fields: text, content, sentence, article
- Minimum 10 character threshold for text samples
- CLI flag: `--dataset-path /path/to/train.jsonl`
