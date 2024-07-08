use num::Float;
use super::super::{ Tensor, TensorType, Vector, VectorType };

pub fn arithmetic<T: Float>(a: &Tensor<T>, b: &Tensor<T>, f: &dyn Fn(&T, &T) -> T) -> Tensor<T> {
    if a.shape == b.shape {
        return arithmetic_same_shape(a, b, f);
    }

    if let TensorType::Scalar(val) = a.get_type() {
        return arithmetic_scalar_tensor(val, b, f);
    }

    if let TensorType::Scalar(val) = b.get_type() {
        return arithmetic_tensor_scalar(a, val, f);
    }

    if let TensorType::Vector(vector_type) = a.get_type() {
        if b.is_matrix() {
            match vector_type {
                VectorType::Bra => return arithmetic_bra_matrix(a, b, f),
                VectorType::Ket => return arithmetic_ket_matrix(a, b, f)
            }
        }
    }
  
    if let TensorType::Vector(vector_type) = b.get_type() {
        if a.is_matrix() {
            match vector_type {
                VectorType::Bra => return arithmetic_matrix_bra(a, b, f),
                VectorType::Ket => return arithmetic_matrix_ket(a, b, f)
            }
        }
    }
    panic!("Unsupported shape {:?} vs {:?}", a.shape, b.shape);
}

fn arithmetic_same_shape<T: Float>(a: &Tensor<T>, b: &Tensor<T>, f: &dyn Fn(&T, &T) -> T) -> Tensor<T> {
    let data = a.data.iter().zip(&b.data).map(|(a, b)| f(a, b)).collect();
    let shape = a.shape.to_vec();
    Tensor { data, shape }
}

fn arithmetic_scalar_tensor<T: Float>(val: T, b: &Tensor<T>, f: &dyn Fn(&T, &T) -> T) -> Tensor<T> {
    let data = b.data.iter().map(|b| f(&val, b)).collect();
    let shape = b.shape.to_vec();
    Tensor { data, shape }
}

fn arithmetic_tensor_scalar<T: Float>(a: &Tensor<T>, val: T, f: &dyn Fn(&T, &T) -> T) -> Tensor<T> {
    let data = a.data.iter().map(|a| f(a, &val)).collect();
    let shape = a.shape.to_vec();
    Tensor { data, shape }
}

fn arithmetic_bra_matrix<T: Float>(a: &Tensor<T>, b: &Tensor<T>, f: &dyn Fn(&T, &T) -> T) -> Tensor<T> {
    assert_eq!(a.col_count(), b.col_count());
    let mut result = Tensor::empty();
    b.rows().for_each(|row| {
        let data = a.data.iter().zip(row.data.iter()).map(|(val_a, val_b)| f(val_a, val_b)).collect();
        result.append_row(Vector::bra(data));
    });
    result 
}

fn arithmetic_ket_matrix<T: Float>(a: &Tensor<T>, b: &Tensor<T>, f: &dyn Fn(&T, &T) -> T) -> Tensor<T> {
    assert_eq!(a.row_count(), b.row_count());
    let mut result = Tensor::empty();
    b.cols().for_each(|col| {
        let data = a.data.iter().zip(col.data.iter()).map(|(val_a, val_b)| f(val_a, val_b)).collect();
        result.append_col(Vector::ket(data));
    });
    result
}

fn arithmetic_matrix_bra<T: Float>(a: &Tensor<T>, b: &Tensor<T>, f: &dyn Fn(&T, &T) -> T) -> Tensor<T> {
    assert_eq!(a.col_count(), b.col_count());
    let mut result = Tensor::empty();
    a.rows().for_each(|row| {
        let data = row.data.iter().zip(b.data.iter()).map(|(val_a, val_b)| f(val_a, val_b)).collect();
        result.append_row(Vector::bra(data));
    });
    result
}

fn arithmetic_matrix_ket<T: Float>(a: &Tensor<T>, b: &Tensor<T>, f: &dyn Fn(&T, &T) -> T) -> Tensor<T> {
    assert_eq!(a.row_count(), b.row_count());
    let mut result = Tensor::empty();
    a.cols().for_each(|col| {
        let data = col.data.iter().zip(b.data.iter()).map(|(val_a, val_b)| f(val_a, val_b)).collect();
        result.append_col(Vector::ket(data));
    });
    result
}
