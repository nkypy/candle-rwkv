# candle-rwkv

### 进度

- [x] RWKV5
- [x] quantized RWKV5
- [ ] RWKV6
- [ ] quantized RWKV6

### 转换格式

```bash
# 下载 rwkv-6-world
cd assets

git lfs install
git clone https://huggingface.co/BlinkDL/rwkv-6-world
# 国内可使用镜像 git clone https://hf-mirror.com/BlinkDL/rwkv-6-world

# pth 转成 safetensors
cargo run --example convert --release -- --input assets/RWKV-5-World-0.4B-v2-20231113-ctx4096.pth

# pth 转成 gguf
cargo run --example quantize --release -- --input assets/RWKV-5-World-0.4B-v2-20231113-ctx4096.pth --config assets/config.json
```

### 运行

```bash
# 示例

# rwkv6
cargo run --example rwkv --release -- --v6 --weight-files converted6.st --config-file assets/config6.json --tokenizer assets/rwkv_vocab_v20230424.json --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"

# rwkv6 量化版
cargo run --example rwkv --release -- --v6 --quantized --weight-files quantized6.gguf --config-file assets/config6.json --tokenizer assets/rwkv_vocab_v20230424.json --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"

# rwkv5
cargo run --example rwkv --release -- --weight-files converted5.st --config-file assets/config5.json --tokenizer assets/rwkv_vocab_v20230424.json --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"

# rwkv5 量化版
cargo run --example rwkv --release -- --quantized --weight-files quantized5.gguf --config-file assets/config5.json --tokenizer assets/rwkv_vocab_v20230424.json --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"
```