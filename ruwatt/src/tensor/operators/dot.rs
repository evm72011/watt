use num::Float;
use std::iter::Sum;
use super::super::{Tensor, IndexTools, TensorType, Scalar};

pub fn dot<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: Float + Sum {
    match (a.get_type(), b.get_type()) {
        (TensorType::Empty, _) | (_, TensorType::Empty) => {
            panic!("Can't dot empty tensors. {:?} . {:?}", a.shape, b.shape)
        },
        (TensorType::Scalar(val_a), TensorType::Scalar(val_b)) => {
            Scalar::new(val_a * val_b)
        },
        (TensorType::Scalar(val_a), _) => {
            b.clone() * val_a
        },
        (_, TensorType::Scalar(val_b)) => {
            a.clone() * val_b
        },
        (TensorType::Vector(_), TensorType::Vector(_)) |
        (TensorType::Vector(_), TensorType::Matrix) |
        (TensorType::Matrix, TensorType::Vector(_)) |
        (TensorType::Matrix, TensorType::Matrix) => {
            dot_matrix(a, b)
        },
        _ => {
            unimplemented!("This method is not yet implemented. {:?} . {:?}", a.shape, b.shape);
        }
    }
}

fn dot_matrix<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: Float + Sum {
    assert_eq!(a.col_count(), b.row_count(), "Incompatible shapes to dot: {:?} vs {:?}", a.shape, b.shape);

    let row_count = a.row_count();
    let col_count = b.col_count();
    let shape = vec![row_count, col_count];
    let mut data = vec![T::zero(); row_count * col_count];

    for row_index in 0..row_count {
        let row = IndexTools::<T>::get_row(row_index, &a.shape, &a.data);
        for col_index in 0..col_count {
            let col = IndexTools::<T>::get_col(col_index, &b.shape, &b.data);
            let value = row.iter().zip(col.iter()).map(|(&a, &b)| a*b).sum();
            data[row_index * row_count + col_index] = value;
        }
    }
    Tensor { shape, data }
}

pub fn _dot<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: Float {
    if a.is_scalar() || b.is_scalar() || a.shape.len() > 2 || a.shape.len() > 2 {
        unimplemented!("This method is not yet implemented");
    }

    assert_eq!(a.shape[1], b.shape[0], "Incompatible shapes to dot: {:?} vs {:?}", a.shape, b.shape);
    let rows = a.row_count();
    let cols = b.col_count();
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
    use super::dot;
    use super::super::super::{Tensor, Matrix, Vector, Scalar};

    #[test]
    #[should_panic(expected = "Can't dot empty tensors. [] . [1, 2]")]
    fn empty_tensor_error() {
        let empty = Tensor::<f32>::empty();
        let vector = Vector::<f32>::bra(vec![1.0, 2.0]);
        dot(&empty, &vector);
    }

    #[test]
    fn vector_scalar() {
        let bra = Vector::bra(vec![ 1.0, 2.0 ]);
        let ket = Scalar::new(2.0);
        let recieved = dot(&bra, &ket);
        let expected = Vector::bra(vec![ 2.0, 4.0 ]);
        assert!(recieved == expected)
    }

    #[test]
    #[should_panic(expected = "Incompatible shapes to dot: [1, 2] vs [1, 2]")]
    fn incompatible_shape_error() {
        let vector = Vector::bra(vec![ 1.0, 2.0 ]);
        dot(&vector, &vector);
    }

    #[test]
    fn dot_vector_vector() {
        let bra = Vector::bra(vec![ 1.0, 2.0 ]);
        let ket = Vector::ket(vec![ 3.0, 4.0 ]);
        let recieved = dot(&bra, &ket);
        let expected = Tensor::new(vec![ 1, 1 ], 11.0);
        assert!(recieved == expected)
    }

    #[test]
    fn dot_matrix_matrix() {
        let a = Matrix::new(vec![
            vec![ 1.0, 2.0, 3.0 ], 
            vec![ 4.0, 5.0, 6.0 ]
        ]);
        let b = Matrix::new(vec![
            vec![ 1.0, 2.0 ], 
            vec![ 3.0, 4.0 ],
            vec![ 5.0, 6.0 ],
        ]);
        let recieved = dot(&a, &b);
        let expected = Matrix::new(vec![
            vec![ 22.0, 28.0 ],
            vec![ 49.0, 64.0 ]
        ]);
        assert_eq!(expected, recieved)
    }
}
