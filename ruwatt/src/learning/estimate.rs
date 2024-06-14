use num::Float;
use std::iter::Sum;
use crate::tensor::Tensor;
use crate::{assert_matrix, assert_shape};

pub fn mse<T>(y_model: Tensor<T>, y_test: Tensor<T>) -> Tensor<T> where T : Float + Sum {
    assert_shape!(y_model, y_test);
    assert_matrix!(y_model);
    let row_count = y_model.row_count();
    if row_count == 0 {
        Tensor::empty()
    } else {
        let data: Vec<T> = y_model.cols().zip(y_test.cols())
            .map(|(col_model, col_test)| {
                let difference = col_model - col_test;
                let summ: T = difference.data.iter().map(|&value| T::powi(value, 2)).sum();
                summ / T::from(row_count).unwrap()
            })
            .collect();
        Tensor::ket(data)
    }
}

pub fn mad<T>(y_model: Tensor<T>, y_test: Tensor<T>) -> Tensor<T> where T : Float + Sum {
    assert_shape!(y_model, y_test);
    assert_matrix!(y_model);
    let row_count = y_model.row_count();
    if row_count == 0 {
        Tensor::empty()
    } else {
        let data: Vec<T> = y_model.cols().zip(y_test.cols())
            .map(|(col_model, col_test)| {
                let difference = col_model - col_test;
                let summ: T = difference.data.iter().map(|&value| T::abs(value,)).sum();
                summ / T::from(row_count).unwrap()
            })
            .collect();
        Tensor::ket(data)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Matrix;

    use super::{mse, mad, Tensor};

    #[test]
    fn mse_test() {
        let y_model = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let y_test = Matrix::new(vec![vec![1.5], vec![1.5], vec![3.5]]);
        let recieved = mse(y_model, y_test);
        let expected = Tensor::ket(vec![0.25]);
        assert_eq!(recieved, expected)
    }

    #[test]
    fn mad_test() {
        let y_model = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let y_test = Matrix::new(vec![vec![1.5], vec![1.5], vec![3.5]]);
        let recieved = mad(y_model, y_test);
        let expected = Tensor::ket(vec![0.5]);
        assert_eq!(recieved, expected)
    }
}