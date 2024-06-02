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

    pub fn random(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data = (0..size).map(|_| T::from(rng.gen::<f32>()).unwrap()).collect();
        Self { shape, data }
    } 
}
