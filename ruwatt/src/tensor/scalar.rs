use num::Float;
use std::marker::PhantomData;
use super::{Tensor, TensorType};
use crate::assert_scalar;

pub struct Scalar<T=f32> {
  _marker: PhantomData<T>
}

impl<T> Scalar<T> where T: Float {
  pub fn new(value: T) -> Tensor<T> {
    Tensor {
        shape: Vec::new(),
        data: vec![value]
    }
  }
}

impl<T> Tensor<T> where T: Float {    

  pub fn is_scalar(&self) -> bool {
    matches!(self.get_type(), TensorType::Scalar(_))
    //self.shape.iter().all(|&value| value == 1)
  }

  pub fn to_scalar(&self) -> T {
    assert_scalar!(self);
    self.data[0]
  }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, Scalar};

    #[test]
    fn scalar() {
        let scalar = Scalar::new(1.0);
        assert_eq!(scalar.shape, Vec::new());
        assert_eq!(scalar.data, vec![1.0]);
    }

    #[test]
    fn is_scalar_true() {
        let scalar = Scalar::new(1.0);
        assert!(scalar.is_scalar());
    }

    #[test]
    fn is_scalar_false() {
        let scalar = Tensor::new(vec![ 1, 2 ], 1.0);
        assert!(!scalar.is_scalar());
    }
}
