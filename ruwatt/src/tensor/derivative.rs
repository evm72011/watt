use num::Float;
use super::Tensor;

pub fn derivative<T>(f: &dyn Fn(&Tensor<T>) -> T, index: usize, tensor: &Tensor<T>) -> T where T: Float {
  let dx = T::from(0.000001).unwrap();
  let mut dw = Tensor::<T>::zeros(tensor.shape.to_vec());
  dw.set(vec![index], dx);
  (f(&(tensor + &dw)) - f(tensor)) / dx
}

pub fn gradient<T>(f: &dyn Fn(&Tensor<T>) -> T, tensor: &Tensor<T>) -> Tensor<T> where T: Float {
  let mut result = Tensor::<T>::zeros(tensor.shape.to_vec());
  for index in 0..tensor.shape[0] {
    let value = derivative(f, index, tensor);
    result.set(vec![index], value);
  }
  result
}
