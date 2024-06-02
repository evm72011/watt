use num::Float;
use super::Tensor;

impl<T> Tensor<T> where T: Float {    
  pub fn scalar(value: T) -> Self {
    Self {
        shape: vec![0],
        data: vec![value]
    }
  }

  pub fn is_scalar(&self) -> bool {
    self.shape == vec![0]
  }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn scalar() {
        let scalar = Tensor::scalar(1.0);
        assert_eq!(scalar.shape, vec![0]);
        assert_eq!(scalar.data, vec![1.0]);
    }

    #[test]
    fn is_scalar_true() {
        let scalar = Tensor::scalar(1.0);
        assert!(scalar.is_scalar());
    }

    #[test]
    fn is_scalar_false() {
        let scalar = Tensor::vector(vec![ 1.0, 2.0 ]);
        assert!(!scalar.is_scalar());
    }
}
