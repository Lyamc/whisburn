use burn::module::{Module, Param};
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module, Debug)]
pub struct ConvRmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    #[module(ignore)]
    pub eps: f64,
}

impl<B: Backend> ConvRmsNorm<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.swap_dims(1, 2);
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        let normed = x * variance.add_scalar(self.eps).powf_scalar(-0.5);
        let gamma = self.weight.val().unsqueeze_dims(&[0, 1]);
        (normed * gamma).swap_dims(1, 2)
    }
}