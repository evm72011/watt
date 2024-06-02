use num::Float;
use super::super::Tensor;

pub fn dot<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: Float {
    assert!(a.shape.len() <= 2);
    assert!(b.shape.len() <= 2);
    assert!(a.shape.len() >= a.shape.len());

    if a.shape.len() == 1 && b.shape.len() == 1 {
        assert_eq!(a.shape[0], b.shape[0]);
        let product = a.data.iter().zip(b.data.iter()).map(|(a,b)| *a * *b).fold(T::zero(), |sum, val| sum + val);
        return Tensor::<T>::vector(vec![product]);
    }

    assert_eq!(a.shape[1], b.shape[0]);
    let rows = a.shape[0];
    let cols = b.shape[1];
    let summs = a.shape[1];
    let mut result = Tensor::zeros(vec![rows, cols]);
    for row in 0..rows {
        for col in 0..cols {
            let mut value = T::zero();
            for i in 0..summs  {
                value = value + *a.get(vec![row, i]).unwrap() * *b.get(vec![i, col]).unwrap();
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
    fn dot_vector_vector() {
        let vector = Tensor::vector(vec![ 1.0, 2.0 ]);
        let recieved = dot(&vector, &vector);
        let expected = Tensor::vector(vec![ 5.0 ]);
        assert!(expected.is_near(recieved))
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
        assert!(expected == recieved)
    }
}