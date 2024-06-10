use crate::{assert_vector, assert_ket, assert_bra};
use num::Float;
use super::Tensor;


impl<T> Tensor<T> where T: Float {    
    pub fn bra(data: Vec<T>) -> Self {
        Self {
            shape: vec![1, data.len()],
            data: data.to_vec()
        }
    }

    pub fn ket(data: Vec<T>) -> Self {
        Self {
            shape: vec![data.len(), 1],
            data: data.to_vec()
        }
    }
  
    pub fn get_v(&self, index: usize) -> &T {
        assert_vector!(self);
        assert!(index < *self.shape.iter().max().unwrap());
        self.data.get(index).unwrap()
    }

    pub fn set_v(&mut self, index: usize, value: T) {
        assert_vector!(self);
        assert!(index < *self.shape.iter().max().unwrap());
        self.data[index] = value
    }

    pub fn ort(is_bra: bool, dim: usize, index: usize, length: T) -> Self {
        if is_bra {
            Self::bra_ort(dim, index, length)
        } else {
            Self::ket_ort(dim, index, length)
        }
    }

    pub fn bra_ort(dim: usize, index: usize, length: T) -> Self {
        let mut result = Tensor::<T>::zeros(vec![1, dim]);
        result.set_v(index, length);
        result
    }

    pub fn ket_ort(dim: usize, index: usize, length: T) -> Self {
        let mut result = Tensor::<T>::zeros(vec![dim, 1]);
        result.set_v(index, length);
        result
    }

    pub fn is_vector(&self) -> bool {
        self.shape.len() == 2 && (self.shape[0] == 1 || self.shape[1] == 1)
    }

    pub fn is_bra(&self) -> bool {
        self.shape.len() == 2 && self.shape[0] == 1
    }

    pub fn is_ket(&self) -> bool {
        self.shape.len() == 2 && self.shape[1] == 1
    }

    
    pub fn dim(&self) -> usize {
        assert_vector!(self);
        *self.shape.iter().max().unwrap()
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

    pub fn to_bra(&self) -> Self{
        assert_ket!(self);
        Tensor::bra(self.data.to_vec())
    }

    pub fn to_ket(&self) -> Self {
        assert_bra!(self);
        Tensor::ket(self.data.to_vec())
    }

    pub fn prepend_one(&self) -> Self {
        let mut data = self.data.to_vec();
        data.insert(0, T::one());
        if self.is_bra() {
            Tensor::bra(data)
        } else {
            Tensor::ket(data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn bra() {
        let vector = Tensor::bra(vec![1.0, 2.0]);
        assert_eq!(vector.shape, vec![1, 2]);
        assert_eq!(vector.data, vec![1.0, 2.0]);
        assert!(vector.is_bra());
        assert!(!vector.is_ket());
    }

    #[test]
    fn ket() {
        let vector = Tensor::ket(vec![1.0, 2.0]);
        assert_eq!(vector.shape, vec![2, 1]);
        assert_eq!(vector.data, vec![1.0, 2.0]);
        assert!(!vector.is_bra());
        assert!(vector.is_ket());
    }

    #[test]
    fn set_get_v() {
        let mut vector = Tensor::bra(vec![1.0, 2.0, 3.0]);
        vector.set_v(1, 4.0);
        let value = vector.get_v(1);
        assert_eq!(*value, 4.0);
    }

    #[test]
    fn ort() {
        let vector = Tensor::ort(true, 3, 1, 2.0);
        assert!(vector.is_bra());
        assert_eq!(vector.shape, vec![1, 3]);
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
    fn dim() {
        let bra = Tensor::bra(vec![1.0, 2.0, 3.0]);
        assert_eq!(bra.dim(), 3);

        let ket = Tensor::ket(vec![1.0, 2.0, 3.0]);
        assert_eq!(ket.dim(), 3);
    }

    #[test]
    fn length() {
        let tensor = Tensor::bra(vec![3.0, 4.0]);
        let length = tensor.length();
        assert_eq!(length, 5.0);
    }
    
    #[test]
    #[should_panic(expected = "Tensor is not a vector: shape = [3, 3]")]
    fn set_length_for_vector_only() {
        let mut tensor = Tensor::new(vec![3, 3], 1.0);
        tensor.set_length(5.0);
    }

    #[test]
    fn to_bra() {
        let tensor = Tensor::ket(vec![3.0, 4.0]);
        let recieved = tensor.to_bra();
        assert!(recieved.is_bra());
    }

    #[test]
    fn to_ket() {
        let tensor = Tensor::bra(vec![3.0, 4.0]);
        let recieved = tensor.to_ket();
        assert!(recieved.is_ket());
    }
}