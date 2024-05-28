use num::Float;
use super::Tensor;

impl<T> Tensor<T> where T: Float {    
  pub fn vector(data: &[T]) -> Self {
    let shape = [data.len()].to_vec();
    Self {
        shape,
        data: data.to_vec()
    }
  }

  pub fn is_vector(&self) -> bool {
    self.shape.len() == 1
  }

  pub fn length(&self) -> T {
      assert!(self.is_vector(), "Must be a vector");
      self.data.iter()
          .map(|&x| x * x)
          .fold(T::zero(), |sum, val| sum + val)
          .sqrt()
  }

  pub fn set_length(&mut self, length: T) {
      assert!(self.is_vector(), "Must be a vector");
      let scale = length / (self.length() + T::min_positive_value());
      for elem in &mut self.data {
          *elem = *elem * scale;
      }
  }
}
