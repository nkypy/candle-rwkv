# candle-rwkv

### TODOs

- [x] RWKV5
- [x] quantized RWKV5
- [x] RWKV6
- [x] quantized RWKV6
- [x] state tuned

### PRs into candle

- RWKV5 [#1743](https://github.com/huggingface/candle/pull/1743)
- RWKV6 [#1781](https://github.com/huggingface/candle/pull/1781)

### Examples

If you just want to have a try. Run command below.

```bash
# run rwkv6
cargo run --release --example rwkv -- --which "world6-3b" --prompt "User: 我在深圳，我要去埃及金字塔，我要怎么走?\n\nAssistant: "

# run quantized rwkv6
cargo run --release --example rwkv -- --quantized --which "world6-3b" --prompt "User: 我在深圳，我要去埃及金字塔，我要怎么走?\n\nAssistant: "

# run state-tuned rwkv6
cargo run --release --example rwkv -- --state-tuned --which "world6-3b" --prompt "我在深圳，我要去埃及金字塔，我要怎么走?"

# run quantized state-tuned rwkv6
cargo run --release --example rwkv -- --quantized --state-tuned --which "world6-3b" --prompt "我在深圳，我要去埃及金字塔，我要怎么走?"
```

If you want to use local model file. First, download pth file from [Hugging Face](https://huggingface.co/BlinkDL). Then run command below.

```bash
# convert pth to safetensors
cargo run --release --example convert -- --input ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth
# run rwkv6
cargo run --release --example rwkv -- --which "world6-1b6" --weight-files ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.safetensors --prompt "User: 我在深圳，我要去埃及金字塔，我要怎么走?\n\nAssistant: "


# quanzited model

# quantize pth to gguf
cargo run --release --example quantize -- --input ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth
# run quantized rwkv6
cargo run --release --example rwkv -- --quantized --which "world6-1b6" --weight-files ./RWKV-x060-World-1B6-v2.1-20240328-ctx4096-q4k.gguf --prompt "User: 我在深圳，我要去埃及金字塔，我要怎么走?\n\nAssistant: "
```

### Others

All PRs are welcome

Powered by [candle](https://github.com/huggingface/candle)