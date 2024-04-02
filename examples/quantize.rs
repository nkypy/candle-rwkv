use core::fmt;
use std::fs::File;

use candle::quantized::{
    gguf_file::{self, Value},
    GgmlDType, QTensor,
};
use clap::{Parser, ValueEnum};
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
    config: Option<String>,
    #[arg(long)]
    output: Option<String>,
    #[arg(long, default_value = "q4k")]
    quantization: Quantization,
}

#[derive(ValueEnum, Debug, Clone)]
enum Quantization {
    #[value(name = "q4_0")]
    Q4_0,
    #[value(name = "q4_1")]
    Q4_1,
    #[value(name = "q5_0")]
    Q5_0,
    #[value(name = "q5_1")]
    Q5_1,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "q8_1")]
    Q8_1,
    Q2k,
    Q3k,
    Q4k,
    Q5k,
    Q6k,
    Q8k,
    F16,
    F32,
}

impl fmt::Display for Quantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let q = match self {
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2k => "q2k",
            Self::Q3k => "q3k",
            Self::Q4k => "q4k",
            Self::Q5k => "q5k",
            Self::Q6k => "q6k",
            Self::Q8k => "q8k",
            Self::F16 => "f16",
            Self::F32 => "f32",
        };
        write!(f, "{}", q)
    }
}

impl Quantization {
    fn dtype(&self) -> GgmlDType {
        match self {
            Self::Q4_0 => GgmlDType::Q4_0,
            Self::Q4_1 => GgmlDType::Q4_1,
            Self::Q5_0 => GgmlDType::Q5_0,
            Self::Q5_1 => GgmlDType::Q5_1,
            Self::Q8_0 => GgmlDType::Q8_0,
            Self::Q8_1 => GgmlDType::Q8_1,
            Self::Q2k => GgmlDType::Q2K,
            Self::Q3k => GgmlDType::Q3K,
            Self::Q4k => GgmlDType::Q4K,
            Self::Q5k => GgmlDType::Q5K,
            Self::Q6k => GgmlDType::Q6K,
            Self::Q8k => GgmlDType::Q8K,
            Self::F16 => GgmlDType::F16,
            Self::F32 => GgmlDType::F32,
        }
    }
}

#[inline]
fn quantize(
    tensors: Vec<(String, candle::Tensor)>,
    dtype: GgmlDType,
) -> anyhow::Result<Vec<(String, QTensor)>> {
    let block_size = dtype.block_size();

    let qtensors = tensors
        .into_iter()
        .map(|(name, tensor)| {
            let should_quantize = tensor.rank() == 2 && tensor.dim(1)? % block_size == 0;
            println!("  quantizing {name} {tensor:?} {should_quantize}");
            let tensor = if should_quantize {
                QTensor::quantize(&tensor, dtype)?
            } else {
                QTensor::quantize(&tensor, GgmlDType::F32)?
            };
            Ok((name, tensor))
        })
        .collect::<candle::Result<Vec<_>>>()?;

    Ok(qtensors)
}

#[inline]
fn metadata(_config: String) -> Vec<(String, Value)> {
    // TODO: add metadata from config file
    // serde_json::from_str(s)
    let metadata = vec![];
    metadata
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // read file to tensors
    println!("reading model file: {}", &args.input);
    let tensors = RepugnantTorchTensors::new_from_file(&args.input)?;

    let file = File::open(args.input)?;
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
            // println!("{}: [{:?}; {:?}]", name, tensor.shape(), tensor.dtype());
            (name, tensor)
        })
        .collect::<Vec<(String, candle::Tensor)>>();
    // save to gguf file
    let qtensors = quantize(tensors, args.quantization.dtype())?;
    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    let metadata = if let Some(config) = args.config {
        metadata(config)
    } else {
        vec![]
    };

    let metadata = metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();

    let output = match &args.output {
        Some(n) => n.to_owned(),
        None => format!("quantized-{}.gguf", args.quantization),
    };
    let mut out_file = std::fs::File::create(&output)?;
    gguf_file::write(&mut out_file, &metadata, &qtensors)?;
    println!("quantized to gguf file: {}", &output);
    Ok(())
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
