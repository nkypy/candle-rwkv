use std::{collections::HashMap, fs::File, path::PathBuf};

use clap::Parser;
use half::{bf16, f16};
use memmap2::Mmap;
use regex::Regex;
use repugnant_pickle::RepugnantTorchTensors;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    output: Option<String>,
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
    // //  time_faaaa -> time_first and reshape
    // if name.ends_with(".time_faaaa") {
    //     name = name.replace(".time_faaaa", ".time_first");
    // }

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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let output = if let Some(output) = args.output {
        output
    } else {
        let mut path = PathBuf::from(&args.input);
        path.set_extension("safetensors");
        path.to_str().unwrap().to_owned()
    };

    // read file to tensors
    println!("converting '{}' to '{}'", &args.input, &output);

    let tensors = RepugnantTorchTensors::new_from_file(&args.input)?;

    let file = File::open(&args.input)?;
    let data = unsafe { Mmap::map(&file)? };

    let tensors = tensors
        .into_iter()
        .map(|x| {
            let name = rename(x.name);

            let start = x.absolute_offset as usize;
            let end = start + x.shape.iter().product::<usize>() * x.tensor_type.size();

            let data: &[bf16] = bytemuck::cast_slice(&data[start..end]);
            let data: Vec<_> = data.iter().map(|x| f16::from_f32(x.to_f32())).collect();

            let tensor = candle::Tensor::from_vec(data, x.shape, &candle::Device::Cpu).unwrap();
            println!("  {}: [{:?}; {:?}]", name, tensor.shape(), tensor.dtype());
            (name, tensor)
        })
        .collect::<HashMap<String, candle::Tensor>>();

    // save to safetensors file
    candle::safetensors::save(&tensors, &output)?;
    println!("converted to safetensors file: {}", output);
    Ok(())
}
