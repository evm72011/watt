use std::ops;
use num::Float;
use super::super::Tensor;
use crate::assert_shape;

impl<T> ops::Div<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: &Tensor<T>) -> Tensor<T>  {
        assert_shape!(self, other);
        let data = self.data.iter().zip(&other.data).map(|(&a, &b)| a / b).collect();
        Tensor {
            data,
            shape: self.shape.to_vec()
        }
    }
}

impl<T> ops::Div<&T> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: &T) -> Tensor<T>  {
        let data = self.data.iter().map(|&a| a / *other).collect();
        Tensor {
            data,
            shape: self.shape.to_vec()
        }
    }
}
