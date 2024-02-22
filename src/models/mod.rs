pub mod quantized;
pub mod rwkv5;
// pub mod rwkv6;

use candle_core::{Result, Tensor};

pub trait RwkvModule {
    type S;

    fn forward(&self, xs: &Tensor, state: &mut Self::S) -> Result<Tensor>;
}
