use std::ops;
use num::Float;
use super::super::Tensor;
use super::arithmetic::arithmetic;

fn sub_tensor_tensor<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    return arithmetic(a, b, &|&x, &y| x - y);
}

impl<T> ops::Sub<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn sub(self, other: &Tensor<T>) -> Self::Output  {
        sub_tensor_tensor(self, other)
  }
}

impl<T> ops::Sub<&Tensor<T>> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn sub(self, other: &Tensor<T>) -> Self::Output  {
        sub_tensor_tensor(&self, other)
  }
}

impl<T> ops::Sub<Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn sub(self, other: Tensor<T>) -> Self::Output  {
        sub_tensor_tensor(self, &other)
  }
}

impl<T> ops::Sub<Tensor<T>> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn sub(self, other: Tensor<T>) -> Self::Output  {
        sub_tensor_tensor(&self, &other)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{Vector, Scalar, Matrix, Tensor};

    fn matrix1234() -> Tensor<f32> {
        Matrix::square(vec![1.0, 2.0, 3.0, 4.0])
    }

    #[test]
    fn sub_same_shape() {
        let a = Vector::bra(vec![ 2.0, 3.0 ]);
        let b = Vector::bra(vec![ 1.0, 2.0 ]);
        let expected = Vector::bra(vec![ 1.0, 1.0 ]);
        assert_eq!(expected, a - b)
    }

    #[test]
    fn scalar_sub_tensor() {
        let a = Scalar::new(2.0);
        let b = Vector::bra(vec![ 1.0, 2.0 ]);
        let expected = Vector::bra(vec![ 1.0, 0.0 ]);
        assert_eq!(expected, a - b)
    }

    #[test]
    fn tensor_sub_scalar() {
        let a = Vector::bra(vec![ 2.0, 3.0 ]);
        let b = Scalar::new(2.0);
        let expected = Vector::bra(vec![ 0.0, 1.0 ]);
        assert_eq!(expected, a - b)
    }

    #[test]
    fn bra_sub_matrix() {
        let a = Vector::bra(vec![ 3.0, 4.0 ]);
        let b = matrix1234();
        let expected =Matrix::new(vec![
            vec![2.0, 2.0],
            vec![0.0, 0.0]
        ]);
        assert_eq!(expected, a - b)
    }

    #[test]
    fn ket_sub_matrix() {
        let a = Vector::ket(vec![ 3.0, 4.0 ]);
        let b = matrix1234();
        let expected =Matrix::new(vec![
            vec![2.0, 1.0],
            vec![1.0, 0.0]
        ]);
        assert_eq!(expected, a - b)
    }

    #[test]
    fn matrix_sub_bra() {
        let a = matrix1234();
        let b = Vector::bra(vec![ 1.0, 2.0 ]);
        let expected =Matrix::new(vec![
            vec![0.0, 0.0],
            vec![2.0, 2.0]
        ]);
        assert_eq!(expected, a - b)
    }

    #[test]
    fn matrix_sub_ket() {
        let a = matrix1234();
        let b = Vector::ket(vec![ 1.0, 2.0 ]);

        let expected =Matrix::new(vec![
            vec![0.0, 1.0],
            vec![1.0, 2.0]
        ]);
        assert_eq!(expected, a - b)
    }
}
