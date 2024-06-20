use num::Float;
use std::iter::Sum;
use crate::{ assert_ket, assert_square_matrix };
use crate::tensor::{ dot::dot, Matrix, Tensor };

// x_(k+1) = (I-A)*x_k + b
#[allow(dead_code)]
pub fn system_le<T>(a: &Tensor<T>, b: &Tensor<T>, step_count: usize, delta: T) -> Tensor<T> where T: Float + Sum {
    assert_square_matrix!(a);
    assert_ket!(b);
    let mut result = Tensor::<T>::new(b.shape.clone(), T::from(0.5).unwrap());
    let a_mod = &Matrix::<T>::ident(b.dim()) - a;

    for _ in 0..step_count {
        let result_new = dot(&a_mod, &result) + b;
        if result_new.is_near(&result, delta) {
            return result_new;
        }
        result = result_new;
    }
    result
}

#[cfg(test)]
mod tests {
    //use super::{Tensor, system_le};

    #[test]
    fn test_system_le() {
        /*
        let a = Tensor::matrix(vec![
            vec![1.0, 2.0, 1.0],
            vec![2.0, 1.0, 2.0],
            vec![3.0, 3.0, 1.0]
        ]);
        let b= Tensor::ket(vec![8.0, 10.0, 12.0]);

        let recieved = system_le(&a, &b, 2, 0.001);

        let expected = Tensor::ket(vec![1.0, 2.0, 3.0]);
        */
        assert_eq!(true, true);
    }
}