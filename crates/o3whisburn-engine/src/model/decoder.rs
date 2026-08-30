use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{backend::Backend, module::embedding, Distribution, Int, Tensor},
};

use crate::model::attention::{
    attn_decoder_mask, MultiHeadCrossAttention, MultiHeadCrossAttentionConfig,
    MultiHeadAttention, MultiHeadAttentionConfig,
};
use crate::model::mlp::{MLP, MLPConfig};

#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
}

impl TextDecoderConfig {
    pub fn init<B: Backend>(&self, tensor_device_ref: &B::Device) -> TextDecoder<B> {
        let token_embedding = Param::from_tensor(Tensor::random(
            [self.n_vocab, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let positional_embedding = Param::from_tensor(Tensor::random(
            [self.n_text_ctx, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let blocks: Vec<_> = (0..self.n_text_layer)
            .into_iter()
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head)
                    .init(tensor_device_ref)
            })
            .collect();
        let ln = nn::LayerNormConfig::new(self.n_text_state).init(tensor_device_ref);

        let mask = Param::from_tensor(attn_decoder_mask(self.n_text_ctx, tensor_device_ref));

        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        TextDecoder {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            n_vocab,
            n_text_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    pub token_embedding: Param<Tensor<B, 2>>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    pub ln: nn::LayerNorm<B>,
    pub mask: Param<Tensor<B, 2>>,
    pub n_vocab: usize,
    pub n_text_ctx: usize,
}

impl<B: Backend> TextDecoder<B> {
    pub fn forward(&self, x: Tensor<B, 2, Int>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.dims();
        let [_, seq_len] = dims;

        assert!(
            seq_len <= self.n_text_ctx,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.n_text_ctx
        );

        let x = embedding(self.token_embedding.val(), x)
            + self
                .positional_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        let mask = self.mask.val().slice([0..seq_len, 0..seq_len]);

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, xa.clone(), mask.clone());
        }

        let x = self.ln.forward(x);
        return x.matmul(self.token_embedding.val().transpose().unsqueeze::<3>());
    }

    pub fn ctx_size(&self) -> usize {
        self.n_text_ctx
    }
}

#[derive(Config)]
pub struct ResidualDecoderAttentionBlockConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl ResidualDecoderAttentionBlockConfig {
    pub fn init<B: Backend>(
        &self,
        tensor_device_ref: &B::Device,
    ) -> ResidualDecoderAttentionBlock<B> {
        let attn =
            MultiHeadAttentionConfig::new(self.n_state, self.n_head).init(tensor_device_ref);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        let cross_attn =
            MultiHeadCrossAttentionConfig::new(self.n_state, self.n_head).init(tensor_device_ref);
        let cross_attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        let mlp = MLPConfig::new(self.n_state).init(tensor_device_ref);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        ResidualDecoderAttentionBlock {
            attn,
            attn_ln,
            cross_attn,
            cross_attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadAttention<B>,
    pub attn_ln: nn::LayerNorm<B>,
    pub cross_attn: MultiHeadCrossAttention<B>,
    pub cross_attn_ln: nn::LayerNorm<B>,
    pub mlp: MLP<B>,
    pub mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        xa: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), Some(mask));
        let x = x.clone() + self.cross_attn.forward(self.cross_attn_ln.forward(x), xa);
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        x
    }
}
