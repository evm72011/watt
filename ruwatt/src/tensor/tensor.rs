use num::Float;
use std::fmt::Debug;
use crate::{ assert_shape, assert_vector, assert_out_of_range };

#[derive(Debug, Clone)]
pub struct Tensor<T = f32> where T: Float {
    pub shape: Vec<usize>,
    pub data: Vec<T>
}

impl<T> Tensor<T> where T: Float {
    pub fn is_small(&self) -> bool {
        let delta = T::from(0.0001).unwrap();
        self.data.iter().any(|x| T::abs(*x) < delta)
    }

    pub fn is_near(&self, other: Self) -> bool {
        let delta = T::from(0.0001).unwrap();
        assert_shape!(self, other);
        self.data.iter().zip(self.data.iter()).any(|(a, b)| T::abs(*a-*b) < delta)
    }

    pub fn get(&self, indices: Vec<usize>) -> T {
        let index = self.calc_index(indices);
        self.data[index]
    }

    pub fn get_v(&self, index: usize) -> &T {
        assert_vector!(self);
        assert!(index < self.shape[0]);
        self.data.get(index).unwrap()
    }

    pub fn set(&mut self, indices: Vec<usize>, value: T) {
        let index = self.calc_index(indices);
        self.data[index] = value;
    }

    fn calc_index(&self, indices: Vec<usize>) -> usize {
        assert_out_of_range!(self, indices);
        let mut index = 0;
        let mut stride = 1;
        for (i, &dim) in self.shape.iter().rev().enumerate() {
            index += indices[self.shape.len() - 1 - i] * stride;
            stride *= dim;
        }
        index
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn is_small() {
        let tensor = Tensor::new(vec![2, 2], 0.00001);
        assert!(tensor.is_small());
    }

    #[test]
    fn is_near() {
        let tensor1 = Tensor::new(vec![2, 2], 0.000001);
        let tensor2 = Tensor::new(vec![2, 2], 0.000001);
        assert!(tensor1.is_near(tensor2));
    }

    #[test]
    fn get_v() {
        let vector = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let value = vector.get_v(2);
        assert_eq!(*value, 3.0);
    }

    #[test]
    fn set_get() {
        let mut matrix = Tensor::zeros(vec![3, 3]);
        matrix.set(vec![1, 1], 5.0);
        let value = matrix.get(vec![1, 1]);
        assert_eq!(value, 5.0);
    }
}