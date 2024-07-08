use num::Float;
use super::super::Tensor;

fn eq_tensors<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
  a.shape == b.shape && a.data == b.data
}

impl<T> PartialEq for Tensor<T> where T: Float {
  fn eq(&self, other: &Self) -> bool {
      eq_tensors(self, other)
  }
}