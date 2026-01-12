use core::fmt;
use std::fs::File;
use std::path::PathBuf;

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
    #[arg(long, default_value = "q4_k_m")]
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
    #[value(name = "q2_k")]
    Q2K,
    #[value(name = "q3_k_s")]
    Q3KS,
    #[value(name = "q3_k_m")]
    Q3KM,
    #[value(name = "q3_k_l")]
    Q3KL,
    #[value(name = "q4_k_s")]
    Q4KS,
    #[value(name = "q4_k_m")]
    Q4KM,
    #[value(name = "q5_k_s")]
    Q5KS,
    #[value(name = "q5_k_m")]
    Q5KM,
    #[value(name = "q6_k")]
    Q6K,
}

impl fmt::Display for Quantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let q = match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q2K => "Q2_K",
            Self::Q3KS => "Q3_K_S",
            Self::Q3KM => "Q3_K_M",
            Self::Q3KL => "Q3_K_L",
            Self::Q4KS => "Q4_K_S",
            Self::Q4KM => "Q4_K_M",
            Self::Q5KS => "Q5_K_S",
            Self::Q5KM => "Q5_K_M",
            Self::Q6K => "Q6_K",
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
            Self::Q8_0 => GgmlDType::Q8_0, // all to Q8_0
            Self::Q2K => GgmlDType::Q2K,
            Self::Q3KS | Self::Q3KM | Self::Q3KL => GgmlDType::Q3K,
            Self::Q4KS | Self::Q4KM => GgmlDType::Q4K,
            Self::Q5KS | Self::Q5KM => GgmlDType::Q5K,
            Self::Q6K => GgmlDType::Q6K, // all to Q6K
        }
    }
}

// "q4_0"    : "Original quant method, 4-bit.",
// "q4_1"    : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
// "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
// "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
// "q8_0"    : "Fast conversion. High resource use, but generally acceptable.",
// "q2_k"    : "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
// "q3_k_s"  : "Uses Q3_K for all tensors",
// "q3_k_m"  : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
// "q3_k_l"  : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
// "q4_k_s"  : "Uses Q4_K for all tensors",
// "q4_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
// "q5_k_s"  : "Uses Q5_K for all tensors",
// "q5_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
// "q6_k"    : "Uses Q6_K for all tensors",

#[inline]
fn quantize(
    tensors: Vec<(String, candle::Tensor)>,
    dtype: GgmlDType,
) -> anyhow::Result<Vec<(String, QTensor)>> {
    let block_size = dtype.block_size();

    let qtensors = tensors
        .into_iter()
        .map(|(name, tensor)| {
            let should_quantize = tensor.rank() >= 2 && tensor.dim(1)? % block_size == 0;
            println!("  quantizing {name} {tensor:?} {should_quantize}");
            let tensor = if should_quantize {
                if name == "head.weight" && dtype != GgmlDType::Q8_0 {
                    QTensor::quantize(&tensor, GgmlDType::Q6K)?
                } else {
                    QTensor::quantize(&tensor, dtype)?
                }
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

    let output = match &args.output {
        Some(n) => n.to_owned(),
        None => {
            let mut path = PathBuf::from(&args.input);
            path.set_extension("");
            path.as_mut_os_string()
                .push(format!(".{}.gguf", args.quantization));
            path.to_str().unwrap().to_owned()
        }
    };

    // read file to tensors
    println!(
        "quantizing '{}' to '{}' as {}",
        &args.input, &output, &args.quantization
    );
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
        name = name.replace(".time_maa_k", ".time_mix_k");
    }
    //  time_maa_v -> time_mix_value and reshape
    if name.ends_with(".time_maa_v") {
        name = name.replace(".time_maa_v", ".time_mix_v");
    }
    //  time_maa_r -> time_mix_receptance and reshape
    if name.ends_with(".time_maa_r") {
        name = name.replace(".time_maa_r", ".time_mix_r");
    }
    //  time_maa_g -> time_mix_gate and reshape
    if name.ends_with(".time_maa_g") {
        name = name.replace(".time_maa_g", ".time_mix_g");
    }
    //  time_maa_w1 -> time_mix_w1 and reshape
    if name.ends_with(".time_maa_w1") {
        name = name.replace(".time_maa_w1", ".time_mix_w1");
    }
    //  time_maa_w2 -> time_mix_w2 and reshape
    if name.ends_with(".time_maa_w2") {
        name = name.replace(".time_maa_w2", ".time_mix_w2");
    }

    //  time_maa_a -> time_mix_a and reshape
    if name.ends_with(".time_maa_a") {
        name = name.replace(".time_maa_a", ".time_mix_a");
    }

    if name != "head.weight" {
        name = "rwkv.".to_owned() + &name
    }

    name
}
