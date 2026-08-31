use burn::tensor::{backend::Backend, Shape, Tensor};
use npy::NpyData;

pub fn numpy_to_tensor<B: Backend, const D: usize>(numpy_data: NpyData<f32>) -> Tensor<B, D> {
    let v: Vec<f32> = numpy_data.to_vec();
    let shape_vec: Vec<usize> = v[0..D]
        .iter()
        .map(|&v| v as usize)
        .collect();
    let shape = Shape::from(shape_vec);

    let device = Default::default();

    Tensor::<B, 1>::from_floats(&v[D..], &device).reshape(shape)
}
