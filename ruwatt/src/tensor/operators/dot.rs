use num::Float;
use super::super::Tensor;

pub fn dot<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: Float {
    if a.is_scalar() || b.is_scalar() || a.shape.len() > 2 || a.shape.len() > 2 {
        unimplemented!("This method is not yet implemented");
    }

    assert_eq!(a.shape[1], b.shape[0], "Incompatible shapes to dot: {:?} vs {:?}", a.shape, b.shape);
    let rows = a.shape[0];
    let cols = b.shape[1];
    let summs = a.shape[1];
    let mut result = Tensor::zeros(vec![rows, cols]);
    for row in 0..rows {
        for col in 0..cols {
            let mut value = T::zero();
            for i in 0..summs  {
                value = value + a.get(vec![row, i]) * b.get(vec![i, col]);
            }
            result.set(vec![row, col], value);
        }
    } 
    result
}

#[cfg(test)]
mod tests {
    use super::{Tensor, dot};

    #[test]
    #[should_panic(expected = "Incompatible shapes to dot: [1, 2] vs [1, 2]")]
    fn dot_error() {
        let vector = Tensor::bra(vec![ 1.0, 2.0 ]);
        dot(&vector, &vector);
    }


    #[test]
    fn dot_vector_vector() {
        let bra = Tensor::bra(vec![ 1.0, 2.0 ]);
        let ket = Tensor::ket(vec![ 3.0, 4.0 ]);
        let recieved = dot(&bra, &ket);
        let expected = Tensor::new(vec![ 1, 1 ], 11.0);
        assert!(recieved == expected)
    }

    #[test]
    fn dot_matrix_matrix() {
        let a = Tensor::matrix(vec![
            vec![ 1.0, 2.0, 3.0 ], 
            vec![ 4.0, 5.0, 6.0 ]
        ]);
        let b = Tensor::matrix(vec![
            vec![ 1.0, 2.0 ], 
            vec![ 3.0, 4.0 ],
            vec![ 5.0, 6.0 ],
        ]);
        let recieved = dot(&a, &b);
        let expected = Tensor::matrix(vec![
            vec![ 22.0, 28.0 ],
            vec![ 49.0, 64.0 ]
        ]);
        assert_eq!(expected, recieved)
    }
}