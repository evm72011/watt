use num::Float;
use std::error::Error;
use std::fmt::Display;
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

pub fn r2_score<T>(y_predict: &Tensor<T>, y_test: &Tensor<T>) -> Tensor<T> where T : Float + Sum{
    assert_shape!(y_predict, y_test);
    assert_matrix!(y_predict);
    let row_count = y_predict.row_count();
    if row_count == 0 {
        return Tensor::empty();
    } 
    let row_means: Vec<T> = y_test.cols()
        .map(|col_test| col_test.data.iter().cloned().sum::<T>() / T::from(col_test.data.len()).unwrap())
        .collect();

    let ss_res: Vec<T> = y_test.cols().zip(y_predict.cols())
        .map(|(col_test, col_predict)| {
            let col_diff = col_test - col_predict;
            let summ: T = col_diff.data.iter().map(|&value| value.powi(2)).sum();
            summ
        })
        .collect();

    let ss_tot: Vec<T> = y_test.cols().zip(row_means)
            .map(|(col_test, mean)| {
                let summ: T = col_test.data.iter().map(|&value| (value - mean).powi(2)).sum();
                summ
            })
            .collect();

    let data = ss_res.iter().zip(ss_tot.iter())
            .map(|(&res, &tot)| T::one() - res / tot)
            .collect();
    Vector::bra(data)
}

pub fn estimate_model<T>(y_predict: &Tensor<T>, y_test: &Tensor<T>, max_mse: T, min_r2: T) -> Result<(), Box<dyn Error>> 
where 
    T : Float + Sum + Display 
{
    let mse_value = mse(&y_predict, &y_test).to_scalar();
    println!("mse = {} (max. {max_mse})", mse_value);
    assert!(mse_value < max_mse);
    let r2_score_value = r2_score(&y_predict, &y_test).to_scalar();
    println!("r2_score = {} (min. {min_r2})", r2_score_value);
    assert!(r2_score_value > min_r2);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{mad, mse, r2_score };
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

    #[test]
    fn r2_score_test() {
        let predict = Vector::ket(vec![2.5, 0.0, 2.0, 8.0]);
        let test = Vector::ket(vec![3.0, -0.5, 2.0, 7.0]);
        let recieved = r2_score(&predict, &test).to_scalar();
        assert!(f64::abs(recieved - 0.9484) < 0.001)
    }
}