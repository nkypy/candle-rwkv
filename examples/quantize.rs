use core::fmt;

use candle::quantized::{
    gguf_file::{self, Value},
    GgmlDType, QTensor,
};
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    config: String,
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
) -> candle::Result<Vec<(String, QTensor)>> {
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

fn main() -> candle::Result<()> {
    let args = Args::parse();

    // read file to tensors
    println!("read safetensors file: {}", args.input);
    let tensors = candle::safetensors::load(args.input, &candle::Device::Cpu)?;
    let mut tensors = tensors.into_iter().map(|(k, v)| (k, v)).collect::<Vec<_>>();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    // save to gguf file
    let qtensors = quantize(tensors, args.quantization.dtype())?;
    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    let metadata = metadata(args.config);
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
