use std::collections::HashMap;

use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use clap::Parser;
use regex::Regex;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    config: String,
    #[arg(long, default_value = "quantized.gguf")]
    output: String,
}

#[inline]
fn rename(mut name: String) -> String {
    //  emb -> embedding
    if name.starts_with("emb.") {
        name = name.replace("emb.", "embeddings.");
    }
    // ln_0 -> pre_ln (only present at block 0)
    if name.starts_with("blocks.0.ln0") {
        name = name.replace("blocks.0.ln0", "blocks.0.pre_ln");
    }
    // att -> attention
    let re = Regex::new(r"blocks\.(\d+)\.att").unwrap();
    name = re.replace(&name, "blocks.$1.attention").to_string();
    // name = re.sub(r"blocks\.(\d+)\.att", r"blocks.\1.attention", name)
    // ffn -> feed_forward
    let re = Regex::new(r"blocks\.(\d+)\.ffn").unwrap();
    name = re.replace(&name, "blocks.$1.feed_forward").to_string();
    // name = re.sub(r"blocks\.(\d+)\.ffn", r"blocks.\1.feed_forward", name)
    // time_mix_k -> time_mix_key and reshape
    if name.ends_with(".time_mix_k") {
        name = name.replace(".time_mix_k", ".time_mix_key");
    }
    // time_mix_v -> time_mix_value and reshape
    if name.ends_with(".time_mix_v") {
        name = name.replace(".time_mix_v", ".time_mix_value");
    }
    // time_mix_r -> time_mix_receptance and reshape
    if name.ends_with(".time_mix_r") {
        name = name.replace(".time_mix_r", ".time_mix_receptance");
    }
    // time_mix_g -> time_mix_gate and reshape
    if name.ends_with(".time_mix_g") {
        name = name.replace(".time_mix_g", ".time_mix_gate");
    }

    //  time_maa_x -> time_mix_x and reshape
    if name.ends_with(".time_maa_x") {
        name = name.replace(".time_maa_x", ".time_mix_x");
    }
    //  time_maa_w -> time_mix_w and reshape
    if name.ends_with(".time_maa_w") {
        name = name.replace(".time_maa_w", ".time_mix_w");
    }
    //  time_maa_k -> time_mix_key and reshape
    if name.ends_with(".time_maa_k") {
        name = name.replace(".time_maa_k", ".time_mix_key");
    }
    //  time_maa_v -> time_mix_value and reshape
    if name.ends_with(".time_maa_v") {
        name = name.replace(".time_maa_v", ".time_mix_value");
    }
    //  time_maa_r -> time_mix_receptance and reshape
    if name.ends_with(".time_maa_r") {
        name = name.replace(".time_maa_r", ".time_mix_receptance");
    }
    //  time_maa_g -> time_mix_gate and reshape
    if name.ends_with(".time_maa_g") {
        name = name.replace(".time_maa_g", ".time_mix_gate");
    }
    //  time_maa_w1 -> time_mix_w1 and reshape
    if name.ends_with(".time_maa_w1") {
        name = name.replace(".time_maa_w1", ".time_mix_w1");
    }
    //  time_maa_w2 -> time_mix_w2 and reshape
    if name.ends_with(".time_maa_w2") {
        name = name.replace(".time_maa_w2", ".time_mix_w2");
    }
    //  time_faaaa -> time_first and reshape
    if name.ends_with(".time_faaaa") {
        name = name.replace(".time_faaaa", ".time_first");
    }

    //  lora_A -> lora.0 and reshape
    if name.ends_with(".lora_A") {
        name = name.replace(".lora_A", ".lora.0");
    }
    // lora_B -> lora.1 and reshape
    if name.ends_with(".lora_B") {
        name = name.replace(".lora_B", ".lora.1");
    }

    if name != "head.weight" {
        name = "rwkv.".to_owned() + &name
    }

    name
}

#[inline]
fn quantize(
    tensors: Vec<(String, candle_core::Tensor)>,
) -> candle_core::Result<Vec<(String, QTensor)>> {
    let dtype = GgmlDType::Q4_0;
    let block_size = dtype.block_size();

    let qtensors = tensors
        .into_iter()
        .map(|(name, tensor)| {
            // TODO:
            // let name = rename(name);
            let should_quantize = tensor.rank() == 2 && tensor.dim(1)? % block_size == 0;
            println!("  quantizing {name} {tensor:?} {should_quantize}");
            let tensor = if should_quantize {
                QTensor::quantize(&tensor, dtype)?
            } else {
                QTensor::quantize(&tensor, GgmlDType::F32)?
            };
            Ok((name, tensor))
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    Ok(qtensors)
}

#[inline]
fn metadata(config: String) -> Vec<(String, Value)> {
    // serde_json::from_str(s)
    let metadata = vec![];
    metadata
}

fn main() -> candle_core::Result<()> {
    let args = Args::parse();

    // read file to tensors
    println!("read pytorch file: {}", args.input);
    let tensors = candle_core::safetensors::load(args.input, &candle_core::Device::Cpu)?;
    let mut tensors = tensors.into_iter().map(|(k, v)| (k, v)).collect::<Vec<_>>();
    // let mut tensors = candle_core::pickle::read_all(args.input)?;
    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    // save to gguf file
    let qtensors = quantize(tensors)?;
    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    let metadata = metadata(args.config);
    let metadata = metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    let mut out_file = std::fs::File::create(&args.output)?;
    gguf_file::write(&mut out_file, &metadata, &qtensors)?;
    println!("quantized to gguf file: {}", args.output);
    Ok(())
}
