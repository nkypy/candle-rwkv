use std::borrow::Cow;
use std::collections::HashMap;

use clap::Parser;
use half::{bf16, f16};
use regex::Regex;
use safetensors::{Dtype, View};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long, default_value = "model.safetensors")]
    output: String,
}

struct Tensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f16>,
}

impl View for Tensor {
    fn dtype(&self) -> Dtype {
        Dtype::F16
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        Cow::Borrowed(bytemuck::cast_slice(&self.data))
    }

    fn data_len(&self) -> usize {
        self.data.len() * self.dtype().size()
    }
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

fn main() -> candle_core::Result<()> {
    let args = Args::parse();

    // read file to tensors
    println!("read pytorch file: {}", args.input);
    let mut tensors = candle_core::pickle::read_all(args.input)?;
    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    // print and change name
    let tensors = tensors
        .into_iter()
        .map(|x| {
            let name = rename(x.0);
            let shape = x.1.shape();
            let size: usize = shape.iter().product();
            let bytes = size * x.1.tensor_type.size();

            println!("{}: [{:?}; {:?}]", name, x.1.shape(), x.1.dtype());
            Tensor { name, shape, data }
            // (
            //     new_name,
            //     x.1.to_dtype(candle_core::DType::F16)
            //         .unwrap()
            //         .contiguous()
            //         .unwrap(),
            // )
        })
        .collect::<Vec<_>>();

    // save to safetensors
    let data = tensors.into_iter().map(|tensor| {
        let name = tensor.name.clone();
        (name, tensor)
    });
    safetensors::serialize_to_file(data, &None, &args.output)?;
    candle_core::safetensors.save()?;
    println!("converted to safetensors file: {}", args.output);
    Ok(())
}
