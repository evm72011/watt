use std::ops;
use num::Float;
use super::super::Tensor;
use crate::assert_shape;

fn mul_tensors<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_shape!(a, b);
    let data = a.data.iter().zip(&b.data).map(|(&a, &b)| a * b).collect();
    Tensor {
        data,
        shape: a.shape.to_vec(),
    }
}

fn mul_tensor_number<T: Float>(a: &Tensor<T>, b: &T) -> Tensor<T> {
    let data = a.data.iter().map(|&a| a * *b).collect();
    Tensor {
        data,
        shape: a.shape.to_vec(),
    }
}

impl<T> ops::Mul<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn mul(self, other: &Tensor<T>) -> Self::Output  {
        mul_tensors(self, other)
    }
}

impl<T> ops::Mul<Tensor<T>> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn mul(self, other: Tensor<T>) -> Self::Output  {
        mul_tensors(&self, &other)
    }
}

impl<T> ops::Mul<&T> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn mul(self, other: &T) -> Self::Output  {
        mul_tensor_number(self, other)
    }
}

impl<T> ops::Mul<&T> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn mul(self, other: &T) -> Self::Output  {
        mul_tensor_number(&self, other)
    }
}

impl<T> ops::Mul<T> for Tensor<T> where T: Float {
  type Output = Tensor<T>;

  fn mul(self, other: T) -> Self::Output  {
    mul_tensor_number(&self, &other)
  }
}

#[cfg(test)]
mod tests {
    use super::super::super::Vector;

    #[test]
    #[should_panic(expected = "Shapes do not match: [1, 2] vs [2, 1]")]
    fn mul_error() {
        let bra = Vector::bra(vec![ 1.0, 2.0 ]);
        let ket = Vector::ket(vec![ 1.0, 2.0 ]);
        let _ = bra * ket;
    }

    #[test]
    fn mul_trensors() {
        let a = Vector::bra(vec![ 1.0, 2.0 ]);
        let b = Vector::bra(vec![ 3.0, 4.0 ]);
        let expected = Vector::bra(vec![ 3.0, 8.0 ]);
        let recieved = a * b;
        assert_eq!(expected, recieved)
    }

    #[test]
    fn mul_trensor_number() {
        let a = Vector::bra(vec![ 1.0, 2.0 ]);
        let expected = Vector::bra(vec![ 2.0, 4.0 ]);
        let recieved = a * 2.0;
        assert_eq!(expected, recieved)
    }
}