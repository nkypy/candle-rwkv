use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use clap::Parser;

use candle_rwkv::models::quantized::rwkv5::{Model as QModel, State as QState};
use candle_rwkv::models::rwkv5::{Config, Model, State, Tokenizer};
use candle_rwkv::models::RwkvModule;

struct TextGeneration<M: RwkvModule> {
    model: M,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl<M: RwkvModule> TextGeneration<M> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: M,
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

    fn run(&mut self, prompt: &str, sample_len: usize, mut state: M::S) -> Result<()> {
        use std::io::Write;
        let mut tokens = self.tokenizer.encode(prompt)?;
        let mut generated_tokens = 0usize;
        let mut next_logits = None;
        for &t in tokens.iter() {
            let input = Tensor::new(&[[t]], &self.device)?;
            let logits = self.model.forward(&input, &mut state)?;
            next_logits = Some(logits);
            print!("{}", self.tokenizer.decode(&[t])?)
        }
        std::io::stdout().flush()?;

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
            print!("{}", self.tokenizer.decode(&[next_token])?);
            std::io::stdout().flush()?;

            let input = Tensor::new(&[[next_token]], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?)
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = true)]
    cpu: bool,

    #[arg(long)]
    v6: bool,

    #[arg(long)]
    quantized: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long, default_value = "1.0")]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value = "0.3")]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 333)]
    sample_len: usize,

    #[arg(long)]
    tokenizer: String,

    #[arg(long)]
    weight_files: String,

    #[arg(long)]
    config_file: String,

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
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(1.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();

    let tokenizer = std::path::PathBuf::from(args.tokenizer);
    let config_filename = std::path::PathBuf::from(args.config_file);
    let filenames = args
        .weight_files
        .split(',')
        .map(std::path::PathBuf::from)
        .collect::<Vec<_>>();
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::new(tokenizer)?;

    let start = std::time::Instant::now();
    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let mut device = Device::Cpu;
    if !args.cpu {
        device = Device::new_cuda(0)?;
    }
    if !args.v6 {
        if !args.quantized {
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)? };
            let model = Model::new(&config, vb)?;
            println!("loaded the model in {:?}", start.elapsed());
            let mut pipeline = TextGeneration::<Model>::new(
                model,
                tokenizer,
                args.seed,
                args.temperature,
                args.top_p,
                args.repeat_penalty,
                args.repeat_last_n,
                &device,
            );
            let state = State::new(1, &config, &device)?;
            pipeline.run(&args.prompt, args.sample_len, state)?;
        } else {
            let vb = QVarBuilder::from_gguf(&args.weight_files, &device)?;
            let model = QModel::new(&config, vb)?;
            println!("loaded the model in {:?}", start.elapsed());
            let mut pipeline = TextGeneration::<QModel>::new(
                model,
                tokenizer,
                args.seed,
                args.temperature,
                args.top_p,
                args.repeat_penalty,
                args.repeat_last_n,
                &device,
            );
            let state = QState::new(1, &config, &device)?;
            pipeline.run(&args.prompt, args.sample_len, state)?;
        }
    } else {
        // TODO: rwkv6
    }
    Ok(())
}
