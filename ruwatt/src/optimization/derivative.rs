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
    let f_2 = T::from(2.0).unwrap();
    for i in 0..size {
        for j in 0..size {
            let value = if i == j {
                let dw = Tensor::<T>::ort(size, i, delta);
                f(&(point + &(&dw * &f_2))) - f_2 * f(&(point + &dw)) + f(point)
            } else {
                let dw_i = Tensor::<T>::ort(size, i, delta);
                let dw_j = Tensor::<T>::ort(size, j, delta);
                f(&(point + &(&dw_i + &dw_j))) - f(&(point + &dw_i)) - f(&(point + &dw_j)) + f(point)
            };
            result.set(vec![i, j], value / (delta * delta));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{Tensor, hessian};

    #[test]
    fn test_hessian_x2_plus_y2() {
        let f = |x: &Tensor::<f32>| x.get(vec![0]).unwrap().powi(2) + x.get(vec![1]).unwrap().powi(2);
        let recieved = hessian(&f, &Tensor::<f32>::zeros(vec![2]), 0.001);
        let mut expected = Tensor::<f32>::identity(2, 2);
        expected = expected * 2.0;
        assert!(expected.is_near(recieved, 0.00001))
    }
}
