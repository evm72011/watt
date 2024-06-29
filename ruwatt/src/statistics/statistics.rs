use num::Float;
use std::marker::PhantomData;
use std::iter::Sum;

use crate::tensor::{Tensor, Vector};
use crate::assert_matrix;

pub struct Statistics<T = f32> {
    _marker: PhantomData<T>
}

impl<T> Statistics<T> where T: Float + Sum {
    pub fn mean(matrix: &Tensor<T>) -> Tensor<T> {
        assert_matrix!(matrix);
        let row_count = T::from(matrix.row_count()).unwrap();
        let data = matrix.cols()
            .map(|col| col.data.iter().cloned().sum::<T>() / row_count)
            .collect();
        Vector::bra(data)
    }

    pub fn std_dev(matrix: &Tensor<T>) -> Tensor<T> {
        assert_matrix!(matrix);
        let means = Self::mean(&matrix);
        let row_count = T::from(matrix.row_count()).unwrap();
        let data = matrix.cols()
            .zip(means.data)
            .map(|(col, mean)| {
                let variance = col.data.iter().cloned()
                    .map(|value| T::powi(value - mean, 2))
                    .sum::<T>() / row_count;
                variance.sqrt()
            })
            .collect();
        Vector::bra(data)
    }
}

#[cfg(test)]
mod tests {
    use super::Statistics;
    use crate::tensor::{Matrix, Vector};
    use crate::assert_near;

    #[test]
    fn mean() {
        let matrix = Matrix::new(vec![
            vec![ 1.0, 2.0 ],
            vec![ 3.0, 4.0 ],
            vec![ 5.0, 6.0 ]
        ]);
        let recieved = Statistics::mean(&matrix);
        let expected = Vector::bra(vec![3.0, 4.0]);
        assert_eq!(recieved, expected);
    }

    #[test]
    fn std_dev() {
        let matrix = Matrix::new(vec![
            vec![ 1.0, 2.0 ], 
            vec![ 3.0, 4.0 ],
            vec![ 5.0, 6.0 ]
        ]);
        let recieved = Statistics::std_dev(&matrix);
        let expected = Vector::bra(vec![1.63, 1.63]);
        assert_near!(recieved, expected, 0.01);
    }
}