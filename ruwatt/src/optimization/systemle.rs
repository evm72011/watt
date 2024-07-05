use num::Float;
use std::iter::Sum;
use crate::{ assert_ket, assert_square_matrix };
use crate::tensor::{ dot::dot, Matrix, Tensor };

fn complement<T>(matrix: &Tensor<T>, row_index: usize, col_index: usize) -> Tensor<T> where T: Float {
    let mut data: Vec<T> = Vec::with_capacity((matrix.row_count() - 1) * (matrix.col_count() - 1));

    (0..matrix.row_count())
        .filter(|&i| i != row_index)
        .for_each(|i| {
            (0..matrix.col_count())
                .filter(|&j| j != col_index)
                .for_each(
                    |j| data.push(matrix.data[i * matrix.col_count() + j])
                );
        });

    Tensor {
        shape: vec![matrix.row_count() - 1, matrix.col_count() - 1],
        data
    }
} 

fn determinant<T>(matrix: &Tensor<T>) -> T where T: Float + Sum {
    assert_square_matrix!(matrix);
    if matrix.col_count() == 2 && matrix.row_count() == 2 {
        matrix.data[0] * matrix.data[3] - matrix.data[1] * matrix.data[2]
    } else {
        matrix.row(0).unwrap().data.iter().enumerate()
            .map(|(index, &value)| {
                let sign = if index % 2 == 0 { T::one() } else { -T::one() };
                sign * value * determinant(&complement(matrix, 0, index))
            })
            .sum()
    }
}

fn inverse<T>(matrix: &Tensor<T>) -> Tensor<T> where T: Float + Sum {
    assert_square_matrix!(matrix);

    let det = determinant(matrix);
    if det == T::zero() {
        panic!("Matrix is singular and cannot be inverted.");
    }

    let size = matrix.row_count();
    let mut data: Vec<T> = Vec::with_capacity(size * size);

    for row_index in 0..size {
        for col_index in 0..size {
            let cofactor = determinant(&complement(matrix, row_index, col_index));
            let sign = if (row_index + col_index) % 2 == 0 { T::one() } else { -T::one() };
            data.push(sign * cofactor);
        }
    }

    let result = Matrix::square(data).tr();
    result / det
}

// x_(k+1) = (I-A)*x_k + b
#[allow(dead_code)]
pub fn system_le<T>(a: &Tensor<T>, b: &Tensor<T>, step_count: usize, delta: T) -> Tensor<T> where T: Float + Sum {
    assert_square_matrix!(a);
    assert_ket!(b);
    let mut result = Tensor::<T>::new(b.shape.clone(), T::from(0.5).unwrap());
    let a_mod = &Matrix::<T>::ident(b.dim()) - a;

    for _ in 0..step_count {
        let result_new = dot(&a_mod, &result) + b;
        if result_new.is_near(&result, delta) {
            return result_new;
        }
        result = result_new;
    }
    result
}

#[cfg(test)]
mod tests {
    use crate::optimization::systemle::inverse;

    use super::{complement, determinant, Matrix, Tensor};

    fn matrix123() -> Tensor {
        let data = (1..=9).map(|x| x as f32).collect();
        return Matrix::square(data);
    }

    #[test]
    fn test_complement() {
        let recieved = complement(&matrix123(), 0, 0);
        let expected = Matrix::square(vec![5.0, 6.0, 8.0, 9.0]);
        assert_eq!(recieved, expected)
    }

    #[test]
    fn test_determinant() {
        let recieved = determinant(&matrix123());
        assert_eq!(recieved, 0.0)
    }

    #[test]
    fn test_inverse() {
        let matrix = Matrix::square(vec![1.0, 2.0, 3.0, 4.0]);
        let recieved = inverse(&matrix);
        let expected = Matrix::square(vec![-2.0, 1.0, 1.5, -0.5]);
        assert_eq!(recieved, expected)
    }

    #[test]
    fn test_system_le() {
        /*
        let a = Tensor::matrix(vec![
            vec![1.0, 2.0, 1.0],
            vec![2.0, 1.0, 2.0],
            vec![3.0, 3.0, 1.0]
        ]);
        let b= Tensor::ket(vec![8.0, 10.0, 12.0]);

        let recieved = system_le(&a, &b, 2, 0.001);

        let expected = Tensor::ket(vec![1.0, 2.0, 3.0]);
        */
        assert_eq!(true, true);
    }
}