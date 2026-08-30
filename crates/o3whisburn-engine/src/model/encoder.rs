use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        self,
        conv::{Conv1d, Conv1dConfig},
        PaddingConfig1d,
    },
    tensor::{backend::Backend, Distribution, Tensor},
};

use crate::model::attention::{MultiHeadAttention, MultiHeadAttentionConfig};
use crate::model::mlp::{MLP, MLPConfig};

#[derive(Config, Debug)]
pub struct AudioEncoderConfig {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
}

impl AudioEncoderConfig {
    pub fn init<B: Backend>(&self, tensor_device_ref: &B::Device) -> AudioEncoder<B> {
        let conv1 = Conv1dConfig::new(self.n_mels, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(tensor_device_ref);
        let gelu1 = nn::Gelu::new();
        let conv2 = Conv1dConfig::new(self.n_audio_state, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_stride(2)
            .init(tensor_device_ref);
        let gelu2 = nn::Gelu::new();
        let blocks: Vec<_> = (0..self.n_audio_layer)
            .into_iter()
            .map(|_| {
                ResidualEncoderAttentionBlockConfig::new(self.n_audio_state, self.n_audio_head)
                    .init(tensor_device_ref)
            })
            .collect();
        let ln_post = nn::LayerNormConfig::new(self.n_audio_state).init(tensor_device_ref);
        let positional_embedding = Param::from_tensor(Tensor::random(
            [self.n_audio_ctx, self.n_audio_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let n_mels = self.n_mels;
        let n_audio_ctx = self.n_audio_ctx;

        AudioEncoder {
            conv1,
            gelu1,
            conv2,
            gelu2,
            blocks,
            ln_post,
            positional_embedding: Some(positional_embedding),
            n_mels,
            n_audio_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub gelu1: nn::Gelu,
    pub conv2: Conv1d<B>,
    pub gelu2: nn::Gelu,
    pub blocks: Vec<ResidualEncoderAttentionBlock<B>>,
    pub ln_post: nn::LayerNorm<B>,
    pub positional_embedding: Option<Param<Tensor<B, 2>>>,
    pub n_mels: usize,
    pub n_audio_ctx: usize,
}

impl<B: Backend> AudioEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.dims();
        let [_, n_mels, _] = dims;

        assert!(
            n_mels == self.n_mels,
            "Audio mel spectrum size must be {}.",
            self.n_mels
        );

        let x = self.gelu1.forward(self.conv1.forward(x));
        let x = self.gelu2.forward(self.conv2.forward(x));

        let x = x.swap_dims(1, 2);
        let k = x.dims()[1];
        
        let mut x = x;
        if let Some(pos_emb) = &self.positional_embedding {
            x = x + pos_emb
                .val()
                .slice([0..k])
                .unsqueeze::<3>();
        }

        for block in &self.blocks {
            x = block.forward(x);
        }

        return self.ln_post.forward(x);
    }

    pub fn ctx_size(&self) -> usize {
        self.n_audio_ctx
    }

    pub fn mel_size(&self) -> usize {
        self.n_mels
    }
}

#[derive(Config)]
pub struct ResidualEncoderAttentionBlockConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl ResidualEncoderAttentionBlockConfig {
    pub fn init<B: Backend>(
        &self,
        tensor_device_ref: &B::Device,
    ) -> ResidualEncoderAttentionBlock<B> {
        let attn =
            MultiHeadAttentionConfig::new(self.n_state, self.n_head).init(tensor_device_ref);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);
        let mlp = MLPConfig::new(self.n_state).init(tensor_device_ref);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        ResidualEncoderAttentionBlock {
            attn,
            attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadAttention<B>,
    pub attn_ln: nn::LayerNorm<B>,
    pub mlp: MLP<B>,
    pub mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualEncoderAttentionBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), None);
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        x
    }
}
