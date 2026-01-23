use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_rwkv::models::rwkv7::{Config, Model, State};
use hf_hub::api::sync::ApiBuilder;

pub fn main() -> Result<()> {
    let api = ApiBuilder::from_env().build().unwrap();
    let repo = api.model("paperfun/rwkv-x070-g1-0b1".to_string());
    let config_filename = repo.get("config.json").unwrap();
    let weights_filename = repo.get("model.safetensors").unwrap();

    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &Device::Cpu)?
    };
    let model = Model::new(&config, vb).unwrap();
    let mut state = State::new(1, &config, None, &Device::Cpu).unwrap();

    let input = Tensor::new(&[[1u32]], &Device::Cpu).unwrap();
    let output = model.forward(&input, &mut state).unwrap();
    let data = output.to_vec3::<f32>().unwrap()[0][0].to_vec();
    println!("Result: {:?}\n====================", &data[..20]);
    let input = Tensor::new(&[[[1f32; 768]]], &Device::Cpu).unwrap();
    let (output, _) = model.blocks[0]
        .attention
        .forward(&input, None, &mut state)
        .unwrap();
    let data = output.to_vec3::<f32>().unwrap()[0][0].to_vec();
    println!(
        "attention Result: {:?}\n=======================",
        &data[..20]
    );
    let output = model.blocks[0]
        .feed_forward
        .forward(&input, &mut state)
        .unwrap();
    let data = output.to_vec3::<f32>().unwrap()[0][0].to_vec();
    println!("feed_forward Result: {:?}", &data[..20]);
    Ok(())
}
