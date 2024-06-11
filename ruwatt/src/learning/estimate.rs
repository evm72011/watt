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