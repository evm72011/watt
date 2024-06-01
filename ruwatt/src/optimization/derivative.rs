use num::Float;
use super::super::Tensor;

pub fn derivative<T>(f: &dyn Fn(&Tensor<T>) -> T, index: usize, point: &Tensor<T>, delta: T) -> T where T: Float {
    let mut dw = Tensor::<T>::zeros(point.shape.to_vec());
    dw.set(vec![index], delta);
    (f(&(point + &dw)) - f(point)) / delta
}

pub fn gradient<T>(f: &dyn Fn(&Tensor<T>) -> T, point: &Tensor<T>, delta: T) -> Tensor<T> where T: Float {
    assert!(point.is_vector(), "Must be a vector");
    let size = point.shape[0];
    let mut result = Tensor::<T>::zeros(vec![size]);
    for i in 0..size {
        let value = derivative(f, i, point, delta);
        result.set(vec![i], value);
    }
    result
}

pub fn hessian<T>(f: &dyn Fn(&Tensor<T>) -> T, point: &Tensor<T>, delta: T) -> Tensor<T> where T: Float {
    assert!(point.is_vector(), "Must be a vector");
    let size = point.shape[0];
    let mut result = Tensor::<T>::zeros(vec![size, size]);
    for i in 0..size {
        for j in 0..size {
            if i == j {

            } else {
                
            }
        }
    }
    result
}