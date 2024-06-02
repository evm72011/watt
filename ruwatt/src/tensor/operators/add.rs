use std::ops;
use num::Float;
use super::super::Tensor;
use crate::assert_shape;

impl<T> ops::Add<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn add(self, other: &Tensor<T>) -> Tensor<T>  {
        assert_shape!(self, other);
        let data = self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect();
        Tensor {
            data,
            shape: self.shape.to_vec()
        }
    }
}

impl<T> ops::Add<Tensor<T>> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn add(self, other: Tensor<T>) -> Tensor<T>  {
        assert_shape!(self, other);
        let data = self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect();
        Tensor {
            data,
            shape: self.shape.to_vec()
        }
    }
}