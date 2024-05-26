use num::Float;
use super::super::Tensor;

pub fn derivative<T>(f: &dyn Fn(&Tensor<T>) -> T, index: usize, point: &Tensor<T>, delta: T) -> T where T: Float {
  let mut dw = Tensor::<T>::zeros(point.shape.to_vec());
  dw.set(vec![index], delta);
  (f(&(point + &dw)) - f(point)) / delta
}

pub fn gradient<T>(f: &dyn Fn(&Tensor<T>) -> T, tensor: &Tensor<T>, delta: T) -> Tensor<T> where T: Float {
  let mut result = Tensor::<T>::zeros(tensor.shape.to_vec());
  for index in 0..tensor.shape[0] {
    let value = derivative(f, index, tensor, delta);
    result.set(vec![index], value);
  }
  result
}
