use num::Float;
use crate::assert_vector;
use crate::tensor::{ Vector,Tensor };

fn derivative<T>(f: &dyn Fn(&Tensor<T>) -> T, index: usize, point: &Tensor<T>, delta: T) -> T where T: Float {
    let dw = Vector::ort(point.is_bra(), point.dim(), index, delta);
    (f(&(point + &dw)) - f(point)) / delta
}

pub fn gradient<T>(f: &dyn Fn(&Tensor<T>) -> T, point: &Tensor<T>, delta: T) -> Tensor<T> where T: Float {
    assert_vector!(point);
    let dim = point.dim();
    let mut result = if point.is_bra() {
        Vector::<T>::bra(vec![T::zero(); dim])
    } else {
        Vector::<T>::ket(vec![T::zero(); dim])
    };
    for i in 0..dim {
        let value = derivative(f, i, point, delta);
        result.set_v(i, value);
    }
    result
}

#[allow(dead_code)]
pub fn hessian<T>(f: &dyn Fn(&Tensor<T>) -> T, point: &Tensor<T>, delta: T) -> Tensor<T> where T: Float {
    assert_vector!(point);
    let dim = point.dim();
    let is_bra = point.is_bra();
    let mut result = Tensor::<T>::zeros(vec![dim, dim]);
    let f_2 = T::from(2.0).unwrap();
    for i in 0..dim {
        for j in 0..dim {
            let value = if i == j {
                let dw = Vector::ort(is_bra, dim, i, delta);
                f(&(point + &(&dw * &f_2))) - f_2 * f(&(point + &dw)) + f(point)
            } else {
                let dw_i = Vector::ort(is_bra, dim, i, delta);
                let dw_j = Vector::ort(is_bra, dim, j, delta);
                f(&(point + &(&dw_i + &dw_j))) - f(&(point + &dw_i)) - f(&(point + &dw_j)) + f(point)
            };
            result.set(vec![i, j], value / (delta * delta));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use num::abs;
    use crate::assert_near;
    use crate::tensor::*;
    use super::{ derivative, gradient, hessian };

    fn f(x: &Tensor) -> f32 {
        x.get_v(0).powi(2) + x.get_v(1).powi(2)
    }

    #[test]
    fn test_derivative() {
        let vector = Vector::ket(vec![1.0, 1.0]);
        let recieved = derivative(&f, 0, &vector, 0.0001);
        assert!(abs(recieved - 2.0) < 0.001)
    }

    #[test]
    fn test_gradient() {
        let vector = Vector::ket(vec![1.0, 1.0]);
        let recieved = gradient(&f, &vector, 0.0001);
        let expected = Vector::ket(vec![2.0, 2.0]);
        assert_near!(expected, recieved, 0.001)
    }

    #[test]
    fn test_hessian() {
        let vector = Vector::ket(vec![0.0, 0.0]);
        let recieved = hessian(&f, &vector, 0.0001);
        let expected = Matrix::new(vec![
            vec![2.0, 0.0],
            vec![0.0, 2.0],
        ]);
        assert_near!(expected, recieved, 0.001)
    }
}
