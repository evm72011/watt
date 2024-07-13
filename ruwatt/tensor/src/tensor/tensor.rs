use num::Float;
use std::fmt::Debug;
use crate::assert_shape;
use super::IndexTools;

#[derive(Debug, Clone)]
pub struct Tensor<T=f64> where T: Float {
    pub shape: Vec<usize>,
    pub data: Vec<T>
}

#[derive(Debug, PartialEq)]
pub enum VectorType {
    Bra,
    Ket
}

#[derive(Debug, PartialEq)]
pub enum TensorType<T> where T: Float {
    Empty,
    Scalar(T),
    Vector(VectorType),
    Matrix,
    General
}

impl<T> PartialEq for Tensor<T> where T: Float {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
  }

impl<T> Tensor<T> where T: Float {
    pub fn get_type(&self) -> TensorType<T> {
        if self.data.len() == 0 && self.shape.len() == 0 {
            TensorType::<T>::Empty
        } else if self.shape.iter().all(|&value| value == 1) {
            TensorType::Scalar(self.data[0])
        } else if self.shape.len() == 2 && self.row_count() == 1 {
            TensorType::Vector(VectorType::Bra)
        } else if self.shape.len() == 2 && self.col_count() == 1 {
            TensorType::Vector(VectorType::Ket)
        } else if self.shape.len() == 2 {
            TensorType::Matrix
        } else {
            TensorType::General
        }
    }

    pub fn is_small(&self, delta: T) -> bool {
        self.data.iter().any(|x| T::abs(*x) < delta)
    }

    pub fn is_empty(&self) -> bool {
        self.data.len() == 0 && self.shape.len() == 0
    }

    pub fn is_near(&self, other: &Self, delta: T) -> bool {
        assert_shape!(self, other);
        self.data.iter().zip(other.data.iter()).all(|(a, b)| T::abs(*a - *b) < delta)
    }

    pub fn get(&self, indices: Vec<usize>) -> T {
        IndexTools::get_item(indices, &self.shape, &self.data)
    }

    pub fn set(&mut self, indices: Vec<usize>, value: T) {
        IndexTools::set_item(indices, value, &self.shape, &mut self.data);
    }

    pub fn assign(&mut self, tensor: Tensor<T>) {
        self.shape = tensor.shape;
        self.data = tensor.data;
    }

    pub fn apply(&mut self, f: impl Fn(T) -> T) {
        self.data = self.data.iter().map(|&x| f(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Tensor, Vector};


    #[test]
    fn is_small() {
        let tensor = Tensor::new(vec![2, 2], 0.00001);
        assert!(tensor.is_small(0.001));
    }

    #[test]
    fn is_empty() {
        let tensor = Tensor::<f64> { shape: vec![], data: vec![] };
        assert!(tensor.is_empty());
    }

    #[test]
    fn is_near_true() {
        let tensor1 = Tensor::new(vec![2, 2], 1.0001);
        let tensor2 = Tensor::new(vec![2, 2], 1.0);
        assert!(tensor1.is_near(&tensor2, 0.001));
    }

    
    #[test]
    fn is_near_false() {
        let tensor1 = Tensor::new(vec![2, 2], 1.002);
        let tensor2 = Tensor::new(vec![2, 2], 1.0);
        assert!(!tensor1.is_near(&tensor2, 0.001));
    }

    #[test]
    fn set_get() {
        let mut matrix = Tensor::zeros(vec![3, 3]);
        matrix.set(vec![1, 1], 5.0);
        let value = matrix.get(vec![1, 1]);
        assert_eq!(value, 5.0);
    }

    #[test]
    fn apply() {
        let mut vector = Vector::bra(vec![1.0, 2.0]);
        vector.apply(|x: f64| x.powi(2));
        let expected = Vector::bra(vec![1.0, 4.0]);
        assert_eq!(vector, expected);
    }
}