use std::ops;
use num::Float;

use super::super::Tensor;
use super::arithmetic::arithmetic;

fn div_tensor_tensor<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    return arithmetic(a, b, &|&x, &y| x / y);
}

impl<T> ops::Div<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: &Tensor<T>) -> Tensor<T>  {
        div_tensor_tensor(self, other)
    }
}

impl<T> ops::Div<&Tensor<T>> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: &Tensor<T>) -> Tensor<T>  {
        div_tensor_tensor(&self, other)
    }
}

impl<T> ops::Div<Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: Tensor<T>) -> Tensor<T>  {
        div_tensor_tensor(self, &other)
    }
}

impl<T> ops::Div<Tensor<T>> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: Tensor<T>) -> Tensor<T>  {
        div_tensor_tensor(&self, &other)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{Vector, Scalar, Matrix, Tensor};

    fn matrix1248() -> Tensor<f32> {
        Matrix::square(vec![1.0, 2.0, 4.0, 8.0])
    }

    #[test]
    fn div_same_shape() {
        let a = Vector::bra(vec![ 4.0, 8.0 ]);
        let b = Vector::bra(vec![ 2.0, 4.0 ]);
        let expected = Vector::bra(vec![ 2.0, 2.0 ]);
        assert_eq!(expected, a / b)
    }

    #[test]
    fn scalar_div_tensor() {
        let a = Scalar::new(2.0);
        let b = Vector::bra(vec![ 2.0, 4.0 ]);
        let expected = Vector::bra(vec![ 1.0, 0.5 ]);
        assert_eq!(expected, a / b)
    }

    #[test]
    fn tensor_div_scalar() {
        let a = Vector::bra(vec![ 2.0, 4.0 ]);
        let b = Scalar::new(2.0);
        let expected = Vector::bra(vec![ 1.0, 2.0 ]);
        assert_eq!(expected, a / b)
    }

    #[test]
    fn bra_div_matrix() {
        let a = Vector::bra(vec![ 4.0, 8.0 ]);
        let b = matrix1248();

        let expected =Matrix::new(vec![
            vec![4.0, 4.0],
            vec![1.0, 1.0]
        ]);
        assert_eq!(expected, a / b)
    }

    #[test]
    fn ket_div_matrix() {
        let a = Vector::ket(vec![ 2.0, 8.0 ]);
        let b = matrix1248();

        let expected =Matrix::new(vec![
            vec![2.0, 1.0],
            vec![2.0, 1.0]
        ]);
        assert_eq!(expected, a / b)
    }

    #[test]
    fn matrix_div_bra() {
        let a = matrix1248();
        let b = Vector::bra(vec![ 1.0, 2.0 ]);

        let expected =Matrix::new(vec![
            vec![1.0, 1.0],
            vec![4.0, 4.0]
        ]);
        assert_eq!(expected, a / b)
    }

    #[test]
    fn matrix_div_ket() {
        let a = matrix1248();
        let b = Vector::ket(vec![ 1.0, 2.0 ]);

        let expected =Matrix::new(vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0]
        ]);
        assert_eq!(expected, a / b)
    }
}
