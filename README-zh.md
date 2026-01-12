# Candle RWKV 🕯️

> **基于 [Candle](https://github.com/huggingface/candle) 框架的 RWKV (Receptance Weighted Key Value) 模型极简、高性能 Rust 实现。**

[![Rust](https://img.shields.io/badge/Built_with-Rust-orange?style=flat-square)](https://www.rust-lang.org/)
[![Candle](https://img.shields.io/badge/Powered_by-Candle-yellow?style=flat-square)](https://github.com/huggingface/candle)
[![License](https://img.shields.io/badge/License-MIT%2FApache-blue?style=flat-square)](LICENSE)

[**Read in English**](README.md)

---

## 🌟 支持模型 (Supported Models)
我们支持 RWKV 家族最新、最强的模型系列：

- ✅ **RWKV7 (Goose)**
- ✅ **RWKV6 (Finch)**
- ✅ **RWKV5 (Eagle)**

## 🚀 快速开始 (Quick Start)

想立即体验？复制下面的命令即可运行。

### 1. 运行 RWKV 模型
直接通过命令行加载并推理。

```bash
# 运行 RWKV7 (Goose)
cargo run --release --example rwkv -- --which "v7-0b1" --prompt "User: 为什么天空是蓝色的？\n\nAssistant: "

# 运行 RWKV6 (Finch)
cargo run --release --example rwkv -- --which "v6-1b6" --prompt "User: 你好，请介绍一下你自己。\n\nAssistant: "
```

### 2. 量化推理 (省显存/内存)
在笔记本上运行？使用量化模式大幅降低内存占用。

```bash
# 运行量化版 RWKV7 (Goose)
cargo run --release --example rwkv -- --quantized --which "v7-0b1" --prompt "User: 给我讲个笑话。\n\nAssistant: "
```

## 🛠️ 进阶用法：本地模型

如果你喜欢自己管理模型文件（例如从 [HuggingFace](https://huggingface.co/BlinkDL) 下载了 `.pth` 权重），我们提供了完整的转换和加载工具。

### 模型转换
首先，需要将 PyTorch 的权重 (`.pth`) 转换为 Rust 原生支持的 SafeTensors 格式，加载速度更快。

```bash
# 转换模型权重
cargo run --release --example convert -- --input ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth

# 转换 State 文件
cargo run --release --example convert -- --input ./rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.pth
```

### 运行本地文件
使用转换后的文件进行推理：

```bash
# 指定本地文件运行
cargo run --release --example rwkv -- \
  --which "v6-1b6" \
  --weight-files ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.safetensors \
  --state-file ./rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.safetensors \
  --prompt "你好，世界！"
```

### 量化工具 (GGUF)
将 `.pth` 文件转换为标准化的 GGUF 格式。

```bash
# 将 .pth 量化为 .gguf
cargo run --release --example quantize -- --input ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth

# 运行本地 GGUF 文件
cargo run --release --example rwkv -- \
  --quantized \
  --which "v6-1b6" \
  --weight-files ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096-q4k.gguf \
  --prompt "User: 你好！\n\nAssistant: "
```

## 🤝 贡献
欢迎提交 PR 或 Issue！

---
*Powered by [candle](https://github.com/huggingface/candle)*
