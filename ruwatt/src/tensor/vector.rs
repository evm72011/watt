use num::Float;
use super::Tensor;

impl<T> Tensor<T> where T: Float {    
  pub fn vector(data: Vec<T>) -> Self {
    let shape = [data.len()].to_vec();
    Self {
        shape,
        data: data.to_vec()
    }
  }

  pub fn ort(dim: usize, index: usize, length: T) -> Self {
    let mut result = Tensor::<T>::zeros(vec![dim]);
    result.set(vec![index], length);
    result
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

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn test_vector() {
        let vector = Tensor::vector(vec![1.0, 1.0]);
        let tensor = Tensor::ones(vec![2]);
        assert_eq!(vector, tensor);
    }

    #[test]
    fn test_is_vector_true() -> Result<(), String> {
        let tensor = Tensor::new(vec![3], 1.0);
        assert_eq!(tensor.is_vector(), true);
        Ok(())
    }

    #[test]
    fn test_is_vector_false() {
        let tensor = Tensor::new(vec![3, 3], 1.0);
        assert_eq!(tensor.is_vector(), false);
    }

    #[test]
    #[should_panic(expected = "Must be a vector")]
    fn test_length_for_vector_only() {
        let tensor = Tensor::new(vec![3, 3], 1.0);
        tensor.length();
    }

    #[test]
    fn test_length() {
        let tensor = Tensor::vector(vec![3.0, 4.0]);
        let length = tensor.length();
        assert_eq!(length, 5.0);
    }

    
    #[test]
    #[should_panic(expected = "Must be a vector")]
    fn test_set_length_for_vector_only() {
        let mut tensor = Tensor::new(vec![3, 3], 1.0);
        tensor.set_length(5.0);
    }
}