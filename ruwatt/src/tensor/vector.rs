use crate::assert_vector;
use num::Float;
use super::Tensor;


impl<T> Tensor<T> where T: Float {    
  pub fn vector(data: Vec<T>) -> Self {
    Self {
        shape: vec![data.len(), 1],
        data: data.to_vec()
    }
  }

  pub fn ort(dim: usize, index: usize, length: T) -> Self {
    let mut result = Tensor::<T>::zeros(vec![dim, 1]);
    result.set(vec![index, 1], length);
    result
  }

  pub fn is_vector(&self) -> bool {
    self.shape.len() == 2 && (self.shape[0] == 1 || self.shape[1] == 1)
  }

  pub fn length(&self) -> T {
      assert_vector!(self);
      self.data.iter()
          .map(|&x| x * x)
          .fold(T::zero(), |sum, val| sum + val)
          .sqrt()
  }

  pub fn set_length(&mut self, length: T) {
    assert_vector!(self);
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
    fn vector() {
        let vector = Tensor::vector(vec![1.0, 1.0]);
        assert_eq!(vector.shape, vec![2, 1]);
        assert_eq!(vector.shape, vec![2, 1]);
    }

    #[test]
    fn ort() {
        let vector = Tensor::ort(3, 1, 2.0);
        assert_eq!(vector.shape, vec![3, 1]);
        assert_eq!(vector.data, vec![0.0, 2.0, 0.0]);
    }

    #[test]
    fn is_vector_true() -> Result<(), String> {
        let tensor = Tensor::new(vec![3, 1], 1.0);
        assert_eq!(tensor.is_vector(), true);
        Ok(())
    }

    #[test]
    fn is_vector_false() {
        let tensor = Tensor::new(vec![3, 2], 1.0);
        assert_eq!(tensor.is_vector(), false);
    }

    #[test]
    #[should_panic(expected = "Tensor is not a vector: shape = [3, 3]")]
    fn length_for_vector_only() {
        let tensor = Tensor::new(vec![3, 3], 1.0);
        tensor.length();
    }

    #[test]
    fn length() {
        let tensor = Tensor::vector(vec![3.0, 4.0]);
        let length = tensor.length();
        assert_eq!(length, 5.0);
    }

    
    #[test]
    #[should_panic(expected = "Tensor is not a vector: shape = [3, 3]")]
    fn set_length_for_vector_only() {
        let mut tensor = Tensor::new(vec![3, 3], 1.0);
        tensor.set_length(5.0);
    }
}