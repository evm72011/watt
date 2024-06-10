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

#[cfg(test)]
mod tests {
    use super::Tensor; 

    #[test]
    fn new() {
        let tensor = Tensor::<f32>::new(vec![2], 1.0);
        assert_eq!(tensor.data, vec![1.0, 1.0]);
    }

    #[test]
    fn zeros() {
        let tensor = Tensor::<f32>::zeros(vec![2]);
        assert_eq!(tensor.data, vec![0.0, 0.0]);
    }

    
    #[test]
    fn ones() {
        let tensor = Tensor::<f32>::ones(vec![2]);
        assert_eq!(tensor.data, vec![1.0, 1.0]);
    }
}