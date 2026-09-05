use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};

#[derive(Config, Debug)]
pub struct SpeechConnectorConfig {
    pub input_dim: usize,
    pub hidden_size: usize,
}

impl SpeechConnectorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpeechConnector<B> {
        SpeechConnector {
            fc1: LinearConfig::new(self.input_dim, self.hidden_size).init(device),
            norm_gamma: Param::from_tensor(Tensor::ones([self.hidden_size], device)),
            fc2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            eps: 1e-6,
        }
    }
}

#[derive(Module, Debug)]
pub struct SpeechConnector<B: Backend> {
    pub fc1: Linear<B>,
    pub norm_gamma: Param<Tensor<B, 1>>,
    pub fc2: Linear<B>,
    #[module(skip)]
    pub eps: f64,
}

impl<B: Backend> SpeechConnector<B> {
    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(features);
        let x = rms_norm(x, self.norm_gamma.val(), self.eps);
        self.fc2.forward(x)
    }
}

fn rms_norm<B: Backend>(x: Tensor<B, 3>, gamma: Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(2);
    let normed = x * variance.add_scalar(eps).powf_scalar(-0.5);
    normed * gamma.unsqueeze_dims(&[0, 1])
}