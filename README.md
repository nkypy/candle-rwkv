# candle-rwkv

### 进度

- [x] RWKV v5
- [ ] quantized RWKV v5
- [x] RWKV v6
- [ ] quantized RWKV v6

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

# v6
cargo run --example rwkv --release -- --weight-files assets/converted.st --config-file assets/config.json --tokenzier assets/rwkv_vocab_v20230424.json --prompt "User: 中国有多少个省？
Assistant: "

# v5 量化版
cargo run --example rwkv --release -- --v5 --quantized --weight-files assets/converted.st --config-file assets/config.json --tokenzier assets/rwkv_vocab_v20230424.json --prompt "User: 中国有多少个省？
Assistant: "
```