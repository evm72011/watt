use num::Float;
use super::Tensor;
use std::fmt::Debug;

impl<T> Tensor<T> where T: Float + Debug {    
    pub fn identity(size: usize) -> Self {
        let mut result = Self::zeros(vec![ size, size ]);
        for i in 0..size {
            result.set(vec![i, i], T::one());
        }
        result
    }

    pub fn matrix(data: Vec<Vec<T>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        let mut result = Tensor::<T>::zeros(vec![rows, cols]);
        for row in 0..rows {
        for col in 0..cols {
            result.set(vec![row, col], data[row][col]);
        }
        }
        result
    }
    }

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn identity() {
        let matrix = Tensor::<f32>::identity(2);
        assert_eq!(matrix.shape, vec![2, 2]);
        assert_eq!(matrix.data, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn matrix() {
        let matrix = Tensor::<f32>::matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(matrix.shape, vec![2, 2]);
        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}