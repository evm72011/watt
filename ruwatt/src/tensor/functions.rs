use std::fmt;
use num::Float;
use super::Tensor;

pub fn sin<T>(tensor: &Tensor<T>) -> Tensor<T> where T: Float + fmt::Display {
  let data = tensor.data.iter().map(|&x| T::sin(x)).collect();
  Tensor {
    data,
    shape: tensor.shape.to_vec()
  }
} 
