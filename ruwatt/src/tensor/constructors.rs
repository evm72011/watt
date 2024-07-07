use num::Float;
use rand::prelude::*;
use super::Tensor;

impl<T> Tensor<T> where T: Float {    
    pub fn new(shape: Vec<usize>, initial_value: T) -> Self {
        let size = shape.iter().product();
        let data = vec![initial_value; size];
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(shape, T::zero())
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        Self::new(shape, T::one())
    }

    pub fn empty() -> Self {
        Self {
            data: vec![],
            shape: vec![]
        }
    }

    pub fn random(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data = (0..size).map(|_| T::from(rng.gen::<f64>()).unwrap()).collect();
        Self { shape, data }
    } 

    pub fn range(start: T, shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = (0..size).map(|x| T::from(x).unwrap() + start).collect();
        Self { shape, data }
    } 
}

#[cfg(test)]
mod tests {
    use crate::tensor::{ Tensor, Vector };

    #[test]
    fn new() {
        let tensor = Tensor::new(vec![2], 1.0);
        assert_eq!(tensor.data, vec![1.0, 1.0]);
    }

    #[test]
    fn zeros() {
        let tensor = Tensor::<f64>::zeros(vec![2]);
        assert_eq!(tensor.data, vec![0.0, 0.0]);
    }
    
    #[test]
    fn ones() {
        let tensor = Tensor::<f64>::ones(vec![2]);
        assert_eq!(tensor.data, vec![1.0, 1.0]);
    }

    #[test]
    fn range() {
        let tensor = Tensor::range(1.0, vec![3, 1]);
        let expected = Vector::ket(vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor, expected);
    }
}