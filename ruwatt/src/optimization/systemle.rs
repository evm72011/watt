use num::Float;
use std::fmt::Debug;
use std::iter::Sum;
use crate::{ assert_ket, assert_square_matrix };
use crate::tensor::{ dot::dot, Tensor };

pub fn solve_system<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: Float + Sum + Debug{
    assert_square_matrix!(a);
    assert_ket!(b);
    dot(&a.inverse(), b)
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Matrix, Vector};
    use super::solve_system;

    #[test]
    fn test_solve_system() {
        let a = Matrix::new(vec![
            vec![1.0, 2.0, 1.0],
            vec![2.0, 1.0, 2.0],
            vec![3.0, 3.0, 1.0]
        ]);
        let b = Vector::ket(vec![8.0, 10.0, 12.0]);
        let recieved = solve_system(&a, &b);
        let expected = Vector::ket(vec![1.0, 2.0, 3.0]);
        assert_eq!(recieved, expected);
    }
}