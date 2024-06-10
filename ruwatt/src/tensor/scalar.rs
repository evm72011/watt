use num::Float;
use super::Tensor;
use crate::assert_scalar;

impl<T> Tensor<T> where T: Float {    
  pub fn scalar(value: T) -> Self {
    Self {
        shape: Vec::new(),
        data: vec![value]
    }
  }

  pub fn is_scalar(&self) -> bool {
    self.shape.iter().all(|&value| value == 1)
  }

  pub fn to_scalar(&self) -> T {
    assert_scalar!(self);
    self.data[0]
  }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn scalar() {
        let scalar = Tensor::scalar(1.0);
        assert_eq!(scalar.shape, Vec::new());
        assert_eq!(scalar.data, vec![1.0]);
    }

    #[test]
    fn is_scalar_true() {
        let scalar = Tensor::scalar(1.0);
        assert!(scalar.is_scalar());
    }

    #[test]
    fn is_scalar_false() {
        let scalar = Tensor::new(vec![ 1, 2 ], 1.0);
        assert!(!scalar.is_scalar());
    }
}
