use num::Float;
use super::super::Tensor;

impl<T> PartialEq for Tensor<T> where T: Float {
  fn eq(&self, other: &Self) -> bool {
      self.shape == other.shape && self.data == other.data
  }
}