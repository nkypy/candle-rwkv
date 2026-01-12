# Candle RWKV 🕯️

> **A minimalist, high-performance implementation of RWKV (Receptance Weighted Key Value) models using [Candle](https://github.com/huggingface/candle) - a lightweight framework for Rust.**

[![Rust](https://img.shields.io/badge/Built_with-Rust-orange?style=flat-square)](https://www.rust-lang.org/)
[![Candle](https://img.shields.io/badge/Powered_by-Candle-yellow?style=flat-square)](https://github.com/huggingface/candle)
[![License](https://img.shields.io/badge/License-MIT%2FApache-blue?style=flat-square)](LICENSE)

[**中文文档 (Chinese)**](README-zh.md)

---

## 🌟 Supported Models
We support the latest and greatest from the RWKV family:

- ✅ **RWKV7 (Goose)**
- ✅ **RWKV6 (Finch)**
- ✅ **RWKV5 (Eagle)**

## 🚀 Quick Start

Ready to run? Here are the commands to get you started immediately.

### 1. Run RWKV Models
Run inference directly from the command line.

```bash
# Run RWKV7 (Goose)
cargo run --release --example rwkv -- --which "g1-1b5" --prompt "User: why is the sky blue?\n\nAssistant: "

# Run RWKV6 (Finch)
cargo run --release --example rwkv -- --which "v6-1b6" --prompt "User: Hello, how are you?\n\nAssistant: "
```

### 2. Quantized Inference (Memory Efficient)
Running on a laptop? Use quantization to save memory.

```bash
# Run Quantized RWKV7 (Goose)
cargo run --release --example rwkv -- --quantized --which "g1-1b5" --prompt "User: Tell me a joke.\n\nAssistant: "
```

## 🛠️ Advanced Usage: Local Models

If you prefer managing your own model files (e.g. download `.pth` from [HuggingFace](https://huggingface.co/BlinkDL)), we provide tools to convert and run them.

### Conversion
First, convert PyTorch weights (`.pth`) to SafeTensors for efficient loading in Rust.

```bash
# Convert Model Weights
cargo run --release --example convert -- --input ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth

# Convert State Files
cargo run --release --example convert -- --input ./rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.pth
```

### Running Local Files
```bash
# Run with local converted files
cargo run --release --example rwkv -- \
  --which "v6-1b6" \
  --weight-files ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.safetensors \
  --state-file ./rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.safetensors \
  --prompt "Hello world!"
```

### Quantization (GGUF)
Convert `.pth` files to standardized GGUF format.

```bash
# Quantize .pth to .gguf
cargo run --release --example quantize -- --input ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth

# Run with local GGUF file
cargo run --release --example rwkv -- \
  --quantized \
  --which "v6-1b6" \
  --weight-files ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096-q4k.gguf \
  --prompt "User: Hello!\n\nAssistant: "
```

## 🤝 Contributing
Contributions are more than welcome! Feel free to open issues or submit PRs.

---
*Powered by [candle](https://github.com/huggingface/candle)*