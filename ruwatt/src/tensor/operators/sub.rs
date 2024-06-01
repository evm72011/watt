use std::ops;
use num::Float;
use super::super::Tensor;

impl<T> ops::Sub<&Tensor<T>> for &Tensor<T> where T: Float {
  type Output = Tensor<T>;

  fn sub(self, other: &Tensor<T>) -> Self::Output  {
      self.compare_shape(&other.shape);
      let data = self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect();
      Tensor {
          data,
          shape: self.shape.to_vec()
      }
  }
}

impl<T> ops::Sub<Tensor<T>> for Tensor<T> where T: Float {
  type Output = Tensor<T>;

  fn sub(self, other: Tensor<T>) -> Self::Output  {
      self.compare_shape(&other.shape);
      let data = self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect();
      Tensor {
          data,
          shape: self.shape.to_vec()
      }
  }
}