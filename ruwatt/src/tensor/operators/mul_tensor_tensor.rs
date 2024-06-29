use std::ops;
use num::Float;

use super::super::Tensor;
use super::arithmetic::arithmetic;

fn mul_tensor_tensor<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    return arithmetic(a, b, &|&a, &b| a * b);
}


impl<T> ops::Mul<&Tensor<T>> for &Tensor<T> where T: Float {
  type Output = Tensor<T>;

  fn mul(self, other: &Tensor<T>) -> Self::Output  {
      mul_tensor_tensor(self, other)
  }
}

impl<T> ops::Mul<&Tensor<T>> for Tensor<T> where T: Float {
  type Output = Tensor<T>;

  fn mul(self, other: &Tensor<T>) -> Self::Output  {
      mul_tensor_tensor(&self, &other)
  }
}

impl<T> ops::Mul<Tensor<T>> for &Tensor<T> where T: Float {
  type Output = Tensor<T>;

  fn mul(self, other: Tensor<T>) -> Self::Output  {
      mul_tensor_tensor(self, &other)
  }
}

impl<T> ops::Mul<Tensor<T>> for Tensor<T> where T: Float {
  type Output = Tensor<T>;

  fn mul(self, other: Tensor<T>) -> Self::Output  {
      mul_tensor_tensor(&self, &other)
  }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Matrix, Tensor};

    use super::super::super::{Vector, Scalar};

    fn matrix1234() -> Tensor<f32> {
        Matrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0]
        ])
    }

    #[test]
    fn mul_same_shape() {
        let a = Vector::bra(vec![ 1.0, 2.0 ]);
        let b = Vector::bra(vec![ 3.0, 4.0 ]);
        let expected = Vector::bra(vec![ 3.0, 8.0 ]);
        assert_eq!(expected, a * b)
    }

    #[test]
    fn mul_scalar_tensor() {
        let a = Scalar::new(2.0);
        let b = Vector::bra(vec![ 3.0, 4.0 ]);
        let expected = Vector::bra(vec![ 6.0, 8.0 ]);
        assert_eq!(expected, a * b)
    }

    #[test]
    fn mul_tensor_scalar() {
        let a = Vector::bra(vec![ 3.0, 4.0 ]);
        let b = Scalar::new(2.0);
        let expected = Vector::bra(vec![ 6.0, 8.0 ]);
        assert_eq!(expected, a * b)
    }

    #[test]
    fn mul_ket_matrix() {
        let a = Vector::ket(vec![ 1.0, 2.0 ]);
        let b = matrix1234();
        let expected =Matrix::new(vec![
            vec![1.0, 2.0],
            vec![6.0, 8.0]
        ]);
        assert_eq!(expected, a * b)
    }

    #[test]
    fn mul_matrix_ket() {
        let a = matrix1234();
        let b = Vector::ket(vec![ 1.0, 2.0 ]);

        let expected =Matrix::new(vec![
            vec![1.0, 2.0],
            vec![6.0, 8.0]
        ]);
        assert_eq!(expected, a * b)
    }

    #[test]
    fn mul_bra_matrix() {
        let a = Vector::bra(vec![ 1.0, 2.0 ]);
        let b = matrix1234();
        let expected =Matrix::new(vec![
            vec![1.0, 4.0],
            vec![3.0, 8.0]
        ]);
        assert_eq!(expected, a * b)
    }

    #[test]
    fn mul_matrix_bra() {
        let a = matrix1234();
        let b = Vector::bra(vec![ 1.0, 2.0 ]);
        let expected =Matrix::new(vec![
            vec![1.0, 4.0],
            vec![3.0, 8.0]
        ]);
        assert_eq!(expected, a * b)
    }
}
