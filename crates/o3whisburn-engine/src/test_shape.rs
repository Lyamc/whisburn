use burn::nn::LinearConfig;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub fn test_linear_shape<B: Backend>(device: &B::Device) {
    let d_in = 4096;
    let d_out = 1024;
    let linear = LinearConfig::new(d_in, d_out).init::<B>(device);
    let weight = linear.weight.val();
    println!("Linear({}, {}) weight shape: {:?}", d_in, d_out, weight.dims());
    
    let x = Tensor::<B, 3>::zeros([1, 192, d_in], device);
    let y = linear.forward(x);
    println!("Output shape: {:?}", y.dims());
}

pub fn test_transpose<B: Backend>(device: &B::Device) {
    let x = Tensor::<B, 2>::zeros([2, 3], device);
    let y = x.clone().transpose();
    println!("Transpose [2, 3] -> {:?}", y.dims());
    
    let x4 = Tensor::<B, 4>::zeros([1, 2, 3, 4], device);
    let y4 = x4.clone().transpose();
    println!("Transpose [1, 2, 3, 4] -> {:?}", y4.dims());
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    #[test]
    fn test_shape() {
        let device = Default::default();
        test_linear_shape::<Wgpu>(&device);
        test_transpose::<Wgpu>(&device);
    }
}
