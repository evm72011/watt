use num::Float;
use super::super::Tensor;

pub fn dot<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: Float {
  assert!(a.shape.len() <= 2);
  assert!(b.shape.len() <= 2);
  assert!(a.shape.len() > a.shape.len());

  if a.shape == vec![1] {
    return b * a.get(vec![0]).unwrap();
  }

  if a.shape.len() == 1 && b.shape.len() ==  1{
    assert_eq!(a.shape[0], b.shape[0]);
    let product = a.data.iter().zip(b.data.iter()).map(|(a,b)| *a * *b).fold(T::zero(), |sum, val| sum + val);
    return Tensor::<T>::vector(vec![product]);
  }

  if a.shape.len() == 2 && b.shape.len() ==  1{
    assert_eq!(a.shape[1], b.shape[0]);
  }

  Tensor::zeros(vec![1])
}