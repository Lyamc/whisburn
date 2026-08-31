use burn::module::{Module, Param};
use burn::nn::{Gelu, Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};

use super::norm::ConvRmsNorm;
use super::sconv::SConv1d;

#[derive(Module, Debug)]
pub struct FfnBlock<B: Backend> {
    pub linear1: Linear<B>,
    pub gelu: Gelu,
    pub linear2: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Block1D<B: Backend> {
    pub norm: ConvRmsNorm<B>,
    pub mixer: SConv1d<B>,
    pub gamma: Param<Tensor<B, 1>>,
    pub ffn_norm: ConvRmsNorm<B>,
    pub ffn: FfnBlock<B>,
    pub ffn_gamma: Param<Tensor<B, 1>>,
}

impl<B: Backend> Block1D<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.norm.forward(x);
        let x = self.mixer.forward(x);
        let x = x * self.gamma.val().unsqueeze_dims(&[0, 2]);
        let x = residual + x;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = x.swap_dims(1, 2);
        let x = self.ffn.linear2.forward(self.ffn.gelu.forward(self.ffn.linear1.forward(x)));
        let x = x.swap_dims(1, 2);
        let x = x * self.ffn_gamma.val().unsqueeze_dims(&[0, 2]);
        residual + x
    }
}

pub fn init_ffn<B: Backend>(dim: usize, ffn_dim: usize, device: &B::Device) -> FfnBlock<B> {
    FfnBlock {
        linear1: LinearConfig::new(dim, ffn_dim).init(device),
        gelu: Gelu::new(),
        linear2: LinearConfig::new(ffn_dim, dim).init(device),
    }
}