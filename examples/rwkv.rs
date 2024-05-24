#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::{Parser, ValueEnum};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};

use candle_rwkv::models::quantized::rwkv5::Model as Q5;
use candle_rwkv::models::quantized::rwkv6::Model as Q6;
use candle_rwkv::models::rwkv5::{Config, Model as M5, State, Tokenizer};
use candle_rwkv::models::rwkv6::Model as M6;

enum Model {
    M5(M5),
    Q5(Q5),
    M6(M6),
    Q6(Q6),
}

impl Model {
    fn forward(&self, xs: &Tensor, state: &mut State) -> candle::Result<Tensor> {
        match self {
            Self::M5(m) => m.forward(xs, state),
            Self::Q5(m) => m.forward(xs, state),
            Self::M6(m) => m.forward(xs, state),
            Self::Q6(m) => m.forward(xs, state),
        }
    }
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, state: &mut State, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        let mut tokens = self.tokenizer.encode(prompt)?;
        let mut generated_tokens = 0usize;
        let mut next_logits = None;
        for &t in tokens.iter() {
            let input = Tensor::new(&[[t]], &self.device)?;
            let logits = self.model.forward(&input, state)?;
            next_logits = Some(logits);
            print!("{}", self.tokenizer.decode(&[t])?)
        }
        std::io::stdout().flush()?;

        let mut last = "".to_string();

        let start_gen = std::time::Instant::now();
        for _ in 0..sample_len {
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => anyhow::bail!("cannot work on an empty prompt"),
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == 261 || next_token == 0 {
                break;
            }
            if next_token == 11 {
                if last.chars().last() == Some('\n') {
                    break;
                };
            }
            let cur = self.tokenizer.decode(&[next_token])?;
            print!("{}", cur);
            last = cur;
            std::io::stdout().flush()?;

            let input = Tensor::new(&[[next_token]], &self.device)?;
            next_logits = Some(self.model.forward(&input, state)?)
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    World1b5,
    World6_1b6,
    World6_3b,
    World6_7b,
}

impl std::fmt::Display for Which {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::World1b5 => "RWKV/rwkv-5-world-1b5",
            _ => "paperfun/rwkv",
        }
    }

    fn revision(&self) -> &'static str {
        match self {
            Self::World1b5 => "refs/pr/2",
            _ => "main",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    #[arg(long, default_value = "world1b5")]
    which: Which,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    state_files: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    quantized: bool,

    #[arg(long)]
    state_tuned: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.which.model_id().to_string(),
        RepoType::Model,
        args.which.revision().to_string(),
    ));
    let tokenizer = api
        .model("lmz/candle-rwkv".to_string())
        .get("rwkv_vocab_v20230424.json")?;
    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => match args.which {
            Which::World1b5 | Which::World6_1b6 => repo.get("config.json")?,
            Which::World6_3b => repo.get("config_3b.json")?,
            Which::World6_7b => repo.get("config_7b.json")?,
        },
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            if args.quantized {
                let file = match args.which {
                    Which::World1b5 => api
                        .model("lmz/candle-rwkv".to_string())
                        .get("world1b5-q4k.gguf")?,
                    Which::World6_1b6 => {
                        repo.get("RWKV-x060-World-1B6-v2.1-20240328-ctx4096-q4k.gguf")?
                    }
                    Which::World6_3b => {
                        repo.get("RWKV-x060-World-3B-v2.1-20240417-ctx4096-q4k.gguf")?
                    }
                    Which::World6_7b => {
                        repo.get("RWKV-x060-World-7B-v2.1-20240507-ctx4096-q4k.gguf")?
                    }
                };
                vec![file]
            } else {
                let file = match args.which {
                    Which::World1b5 => repo.get("model.safetensors")?,
                    Which::World6_1b6 => {
                        repo.get("RWKV-x060-World-1B6-v2.1-20240328-ctx4096.safetensors")?
                    }
                    Which::World6_3b => {
                        repo.get("RWKV-x060-World-3B-v2.1-20240417-ctx4096.safetensors")?
                    }
                    Which::World6_7b => {
                        repo.get("RWKV-x060-World-7B-v2.1-20240507-ctx4096.safetensors")?
                    }
                };
                vec![file]
            }
        }
    };
    println!("retrieved the weight files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::new(tokenizer)?;

    let start = std::time::Instant::now();
    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let device = if !args.cpu && candle::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else if !args.cpu && candle::utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    let model = if args.quantized {
        let filename = &filenames[0];
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename, &device)?;
        match args.which {
            Which::World1b5 => Model::Q5(Q5::new(&config, vb)?),
            _ => Model::Q6(Q6::new(&config, vb)?),
        }
    } else {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)? };
        match args.which {
            Which::World1b5 => Model::M5(M5::new(&config, vb)?),
            _ => Model::M6(M6::new(&config, vb)?),
        }
    };
    println!("loaded the model on {:?} in {:?}", &device, start.elapsed());

    let start = std::time::Instant::now();
    let state_file = match args.state_files {
        Some(files) => Some(files.into()),
        None => {
            if args.state_tuned {
                match args.which {
                    Which::World6_1b6 => Some(
                        repo.get("rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.safetensors")?,
                    ),
                    Which::World6_3b => Some(
                        repo.get("rwkv-x060-chn_single_round_qa-3B-20240516-ctx2048.safetensors")?,
                    ),
                    Which::World6_7b => Some(
                        repo.get("rwkv-x060-chn_single_round_qa-7B-20240516-ctx2048.safetensors")?,
                    ),
                    _ => None,
                }
            } else {
                None
            }
        }
    };

    if !state_file.is_none() {
        println!("retrieved the state files in {:?}", start.elapsed());
    }

    // state files are small, no need to quantize
    let mut state = match state_file {
        Some(file) => {
            let start = std::time::Instant::now();
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&vec![file], DType::F32, &device)? };
            println!("loaded the state on {:?} in {:?}", &device, start.elapsed());
            State::new(1, &config, Some(vb), &device)?
        }
        None => State::new(1, &config, None, &device)?,
    };

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    let prompt = if args.state_tuned {
        format!("User: {}\n\nAssistant: ", args.prompt.clone())
    } else {
        args.prompt.clone()
    };
    pipeline.run(&mut state, &prompt, args.sample_len)?;
    Ok(())
}
