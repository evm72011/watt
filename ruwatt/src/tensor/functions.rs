use num::Float;
use super::Tensor;

#[allow(dead_code)]
pub fn abs<T>(tensor: &Tensor<T>) -> Tensor<T> where T: Float {
  let data = tensor.data.iter().map(|&x| T::abs(x)).collect();
  Tensor {
    data,
    shape: tensor.shape.to_vec()
  }
} 
