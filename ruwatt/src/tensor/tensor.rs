use num::Float;
use std::fmt::Debug;
use crate::assert_shape;

use super::index_tools::IndexTools;

#[derive(Debug, Clone)]
pub struct Tensor<T = f32> where T: Float {
    pub shape: Vec<usize>,
    pub data: Vec<T>
}

impl<T> Tensor<T> where T: Float {
    pub fn is_small(&self, delta: T) -> bool {
        self.data.iter().any(|x| T::abs(*x) < delta)
    }

    pub fn is_near(&self, other: &Self, delta: T) -> bool {
        assert_shape!(self, other);
        self.data.iter().zip(other.data.iter()).all(|(a, b)| T::abs(*a - *b) < delta)
    }

    pub fn get(&self, indices: Vec<usize>) -> T {
        IndexTools::get_item(indices, &self.shape, &self.data)
        //let index = self.calc_index(indices);
        //self.data[index]
    }

    pub fn set(&mut self, indices: Vec<usize>, value: T) {
        IndexTools::set_item(indices, value, &self.shape, &mut self.data);
        //let index = self.calc_index(indices);
        //self.data[index] = value;
    }

    /*
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
    */
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn is_small() {
        let tensor = Tensor::new(vec![2, 2], 0.00001);
        assert!(tensor.is_small(0.001));
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
}