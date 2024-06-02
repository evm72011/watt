use num::Float;
use super::super::Tensor;
use crate::{ assert_ket, assert_square_matrix, tensor::dot::dot };

fn zero_diagonal<T>(matrix: &Tensor<T>) -> Tensor<T> where T: Float {
    let dim = matrix.shape[0];
    let mut result = matrix.clone();
    for i in 0..dim {
        result.set(vec![i , i], T::zero())
    }
    result
}

fn vector_diagonal<T>(matrix: &Tensor<T>) -> Tensor<T> where T: Float {
    let dim = matrix.shape[0];
    let mut result = Tensor::<T>::zeros(vec![dim, 1]);
    for i in 0..dim {
        let value = matrix.get(vec![i , i]);
        result.set_v(i, value)
    }
    result
}

pub fn gauss<T>(a: &Tensor<T>, b:&Tensor<T>, step_count: usize) -> Tensor<T> where T: Float {
    assert_square_matrix!(a);
    assert_ket!(b);
    let mut result = b.clone();
    let a_mod = zero_diagonal(a);
    let diagonal = vector_diagonal(a);

    for _ in 0..step_count {
        result = b - &dot(&a_mod, &result);
        for i in 0..b.dim() {
            let value = result.get_v(i) / diagonal.get_v(i);
            result.set_v(i, value)
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{Tensor, zero_diagonal, vector_diagonal};

    #[test]
    fn test_zero_diagonal() {
        let matrix = Tensor::matrix(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let recieved = zero_diagonal(&matrix);
        let expected = Tensor::matrix(vec![
            vec![0.0, 2.0],
            vec![3.0, 0.0],
        ]);
        assert_eq!(recieved, expected);
    }

    #[test]
    fn test_vector_diagonal() {
        let matrix = Tensor::matrix(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let recieved = vector_diagonal(&matrix);
        let expected = Tensor::ket(vec![1.0, 4.0]);
        assert_eq!(recieved, expected);
    }
}