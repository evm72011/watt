use num::Float;
use std::iter::Sum;
use crate::tensor::{Tensor, Vector};
use crate::{assert_matrix, assert_shape};

pub fn mse<T>(y_predict: &Tensor<T>, y_test: &Tensor<T>) -> Tensor<T> where T : Float + Sum {
    assert_shape!(y_predict, y_test);
    assert_matrix!(y_predict);
    let row_count = y_predict.row_count();
    if row_count == 0 {
        Tensor::empty()
    } else {
        let data: Vec<T> = y_predict.cols().zip(y_test.cols())
            .map(|(col_model, col_test)| {
                let difference = col_model - col_test;
                let summ: T = difference.data.iter().map(|&value| T::powi(value, 2)).sum();
                summ / T::from(row_count).unwrap()
            })
            .collect();
        Vector::ket(data)
    }
}

pub fn mad<T>(y_predict: &Tensor<T>, y_test: &Tensor<T>) -> Tensor<T> where T : Float + Sum {
    assert_shape!(y_predict, y_test);
    assert_matrix!(y_predict);
    let row_count = y_predict.row_count();
    if row_count == 0 {
        Tensor::empty()
    } else {
        let data: Vec<T> = y_predict.cols().zip(y_test.cols())
            .map(|(col_model, col_test)| {
                let difference = col_model - col_test;
                let summ: T = difference.data.iter().map(|&value| T::abs(value,)).sum();
                summ / T::from(row_count).unwrap()
            })
            .collect();
        Vector::ket(data)
    }
}

/*
pub fn r_score<T>(y_predict: &Tensor<T>, y_test: &Tensor<T>) -> Tensor<T> where T : Float + Sum {
    assert_shape!(y_predict, y_test);
    assert_matrix!(y_predict);
    let row_count = y_predict.row_count();
    if row_count == 0 {
        Tensor::empty()
    } else {
        let means = y_test.cols()
            .map(|col| col.data.iter().sum() / col.data.len());

        let ssrs: Vec<T> = y_predict.cols().zip(y_test.cols())
            .map(|(col_model, col_test)| {
                let difference = col_model - col_test;
                let summ: T = difference.data.iter().map(|&value| value.powi(2)).sum();
                summ / T::from(row_count).unwrap()
            })
            .collect();

        let ssrt: Vec<T> = y_test.cols().zip(means)
            .map(|(col_test, mean)| {
                let difference = col_test - mean;
                let summ: T = difference.data.iter().map(|&value| value.powi(2)).sum();
                summ / T::from(row_count).unwrap()
            })
            .collect();

        //let data = 
        Vector::ket(data)
    }
}
*/

#[cfg(test)]
mod tests {
    use super::{mse, mad };
    use crate::tensor::{Vector, Matrix};

    #[test]
    fn mse_test() {
        let y_model = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let y_test = Matrix::new(vec![vec![1.5], vec![1.5], vec![3.5]]);
        let recieved = mse(&y_model, &y_test);
        let expected = Vector::ket(vec![0.25]);
        assert_eq!(recieved, expected)
    }

    #[test]
    fn mad_test() {
        let y_model = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let y_test = Matrix::new(vec![vec![1.5], vec![1.5], vec![3.5]]);
        let recieved = mad(&y_model, &y_test);
        let expected = Vector::ket(vec![0.5]);
        assert_eq!(recieved, expected)
    }
}