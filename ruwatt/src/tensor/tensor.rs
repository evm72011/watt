use num::Float;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct Tensor<T = f32> where T: Float {
    pub shape: Vec<usize>,
    pub data: Vec<T>
}

impl<T> Tensor<T> where T: Float {
    pub fn compare_shape(&self, shape: &Vec<usize>) {
        assert_eq!(self.shape, *shape, "Shape mismatch");
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn is_small(&self, delta: T) -> bool {
        self.data.iter().any(|x| T::abs(*x) < delta)
    }

    pub fn is_near(&self, other: Self, delta: T) -> bool {
        self.compare_shape(&other.shape);
        self.data.iter().zip(self.data.iter()).any(|(a,b)| T::abs(*a-*b) < delta)
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn get(&self, indices: Vec<usize>) -> Option<&T> {
        self.calculate_index(indices).and_then(|idx| self.data.get(idx))
    }

    pub fn set(&mut self, indices: Vec<usize>, value: T) -> Option<()> {
        if let Some(idx) = self.calculate_index(indices) {
            if let Some(elem) = self.data.get_mut(idx) {
                *elem = value;
                return Some(());
            }
        }
        None
    }

    fn calculate_index(&self, indices: Vec<usize>) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut index = 0;
        let mut stride = 1;
        for (i, &dim) in self.shape.iter().rev().enumerate() {
            if indices[self.shape.len() - 1 - i] >= dim {
                return None;
            }
            index += indices[self.shape.len() - 1 - i] * stride;
            stride *= dim;
        }
        Some(index)
    }
}
