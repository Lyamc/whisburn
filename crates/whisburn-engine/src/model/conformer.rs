use burn::{
    config::Config,
    module::{Module, Param},
    nn::{conv::{Conv1d, Conv1dConfig}, Linear, LinearConfig, LayerNorm, LayerNormConfig, PaddingConfig1d, BatchNorm, BatchNormConfig},
    tensor::{backend::Backend, Tensor},
};
use crate::model::attention::{RelPosMultiHeadAttention, RelPosMultiHeadAttentionConfig};

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    #[module(ignore)]
    pub epsilon: f64,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Self {
        Self::with_eps(d_model, 1e-5, device)
    }

    pub fn with_eps(d_model: usize, epsilon: f64, device: &B::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([d_model], device)),
            epsilon,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let rms = (x.clone().powf_scalar(2.0).mean_dim(D - 1) + self.epsilon).sqrt();
        (x / rms) * self.gamma.val().unsqueeze()
    }
}

#[derive(Config, Debug)]
pub struct ConformerEncoderConfig {
    pub d_model: usize,
    pub n_head: usize,
    pub n_layers: usize,
    pub kernel_size: usize,
}

impl ConformerEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConformerEncoder<B> {
        let layers = (0..self.n_layers)
            .map(|_| ConformerBlockConfig::new(self.d_model, self.n_head, self.kernel_size).init(device))
            .collect();
        ConformerEncoder { layers }
    }
}

#[derive(Module, Debug)]
pub struct ConformerEncoder<B: Backend> {
    pub layers: Vec<ConformerBlock<B>>,
}

impl<B: Backend> ConformerEncoder<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Config, Debug)]
pub struct ConformerBlockConfig {
    pub d_model: usize,
    pub n_head: usize,
    pub kernel_size: usize,
}

impl ConformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConformerBlock<B> {
        let ff1 = FeedForwardModuleConfig::new(self.d_model).init(device);
        let attn = RelPosMultiHeadAttentionConfig::new(self.d_model, self.n_head).init(device);
        let attn_ln = LayerNormConfig::new(self.d_model).init(device);
        let conv = ConvolutionModuleConfig::new(self.d_model, self.kernel_size).init(device);
        let ff2 = FeedForwardModuleConfig::new(self.d_model).init(device);
        let final_ln = LayerNormConfig::new(self.d_model).init(device);

        ConformerBlock {
            ff1,
            attn,
            attn_ln,
            conv,
            ff2,
            final_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ConformerBlock<B: Backend> {
    pub ff1: FeedForwardModule<B>,
    pub attn: RelPosMultiHeadAttention<B>,
    pub attn_ln: LayerNorm<B>,
    pub conv: ConvolutionModule<B>,
    pub ff2: FeedForwardModule<B>,
    pub final_ln: LayerNorm<B>,
}

impl<B: Backend> ConformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.ff1.forward(x.clone()).mul_scalar(0.5);
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x.clone()), None);
        let x = x.clone() + self.conv.forward(x.clone());
        let x = x.clone() + self.ff2.forward(x).mul_scalar(0.5);
        self.final_ln.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct FeedForwardModuleConfig {
    pub d_model: usize,
}

impl FeedForwardModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForwardModule<B> {
        let ln = LayerNormConfig::new(self.d_model).init(device);
        let lin1 = LinearConfig::new(self.d_model, self.d_model * 4).init(device);
        let lin2 = LinearConfig::new(self.d_model * 4, self.d_model).init(device);
        FeedForwardModule { ln, lin1, lin2 }
    }
}

#[derive(Module, Debug)]
pub struct FeedForwardModule<B: Backend> {
    pub ln: LayerNorm<B>,
    pub lin1: Linear<B>,
    pub lin2: Linear<B>,
}

impl<B: Backend> FeedForwardModule<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.ln.forward(x);
        
        // Manual lin1
        let x = x.matmul(self.lin1.weight.val().transpose().unsqueeze::<3>());
        let x = if let Some(bias) = &self.lin1.bias {
            x + bias.val().unsqueeze::<3>()
        } else {
            x
        };
        
        let x = burn::tensor::activation::silu(x);
        
        // Manual lin2
        let x = x.matmul(self.lin2.weight.val().transpose().unsqueeze::<3>());
        if let Some(bias) = &self.lin2.bias {
            x + bias.val().unsqueeze::<3>()
        } else {
            x
        }
    }
}

#[derive(Config, Debug)]
pub struct ConvolutionModuleConfig {
    pub d_model: usize,
    pub kernel_size: usize,
}

impl ConvolutionModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvolutionModule<B> {
        let ln = LayerNormConfig::new(self.d_model).init(device);
        let point_conv1 = Conv1dConfig::new(self.d_model, self.d_model * 2, 1).init(device);
        let depth_conv = Conv1dConfig::new(self.d_model, self.d_model, self.kernel_size)
            .with_groups(self.d_model)
            .with_padding(PaddingConfig1d::Explicit(self.kernel_size / 2))
            .init(device);
        let bn = BatchNormConfig::new(self.d_model).init(device);
        let point_conv2 = Conv1dConfig::new(self.d_model, self.d_model, 1).init(device);
        
        ConvolutionModule { ln, point_conv1, depth_conv, bn, point_conv2 }
    }
}

#[derive(Module, Debug)]
pub struct ConvolutionModule<B: Backend> {
    pub ln: LayerNorm<B>,
    pub point_conv1: Conv1d<B>,
    pub depth_conv: Conv1d<B>,
    pub bn: BatchNorm<B, 1>,
    pub point_conv2: Conv1d<B>,
}

impl<B: Backend> ConvolutionModule<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.ln.forward(x);
        let x = x.swap_dims(1, 2);
        
        let x = self.point_conv1.forward(x);
        let [b, d2, s] = x.dims();
        let d = d2 / 2;
        let x1 = x.clone().slice([0..b, 0..d, 0..s]);
        let x2 = x.slice([0..b, d..d2, 0..s]);
        let x = x1 * burn::tensor::activation::sigmoid(x2);
        
        let x = self.depth_conv.forward(x);
        let x = self.bn.forward(x);
        let x = burn::tensor::activation::silu(x);
        let x = self.point_conv2.forward(x);
        
        x.swap_dims(1, 2)
    }
}
