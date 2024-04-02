# candle-rwkv

### TODOs

- [x] RWKV5
- [x] quantized RWKV5
- [x] RWKV6
- [x] quantized RWKV6
- [ ] RWKV7
- [ ] quantized RWKV7

### PRs

- RWKV5 [#1743](https://github.com/huggingface/candle/pull/1743)
- RWKV6 [#1781](https://github.com/huggingface/candle/pull/1781)

### Examples

```bash
# convert pth to safetensors
cargo run --release --example convert -- --input RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth

# quantize pth to gguf
cargo run --release --example quantize -- --input RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth

# rwkv6
cargo run --example rwkv --release -- --which "world1b6" --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"

# quantized rwkv6
cargo run --example rwkv --release -- --quantized --which "world1b6" --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"

# rwkv5
cargo run --example rwkv --release -- --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"

# quantized rwkv5
cargo run --example rwkv --release -- --quantized --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"
```

### Others

All PRs are welcome

Powered by [candle](https://github.com/huggingface/candle)