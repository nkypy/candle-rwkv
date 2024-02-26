# candle-rwkv

### 进度

- [x] RWKV5
- [x] quantized RWKV5
- [ ] RWKV6
- [ ] quantized RWKV6

### 运行

```bash
# 示例

# rwkv6
cargo run --example rwkv --release -- --which "world1b6" --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"

# rwkv6 量化版
cargo run --example rwkv --release -- --quantized --which "world1b6" --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"

# rwkv5
cargo run --example rwkv --release -- --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"

# rwkv5 量化版
cargo run --example rwkv --release -- --quantized --prompt "Assistant: Sure! Here is a very detailed plan to create flying pigs:"
```