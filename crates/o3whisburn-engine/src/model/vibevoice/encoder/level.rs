use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor};

use super::block::Block1D;
use super::sconv::SConv1d;

#[derive(Module, Debug)]
pub struct EncoderLevel<B: Backend> {
    pub conv: SConv1d<B>,
    pub blocks: Vec<Block1D<B>>,
}

impl<B: Backend> EncoderLevel<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv.forward(x);
        self.blocks.iter().fold(x, |acc, block| block.forward(acc))
    }
}