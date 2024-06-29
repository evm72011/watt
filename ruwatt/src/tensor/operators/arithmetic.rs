use num::Float;
use crate::tensor::{TensorType, VectorType};
use super::super::Tensor;

pub fn arithmetic<T: Float>(a: &Tensor<T>, b: &Tensor<T>, f: &dyn Fn(&T, &T) -> T) -> Tensor<T> {
  if a.shape == b.shape {
      let data = a.data.iter().zip(&b.data).map(|(a, b)| f(a, b)).collect();
      let shape = a.shape.to_vec();
      return Tensor { data, shape};
  }

  if let TensorType::Scalar(val) = a.get_type() {
      let data = b.data.iter().map(|b| f(&val, b)).collect();
      let shape = b.shape.to_vec();
      return Tensor { data, shape};
  }

  if let TensorType::Scalar(_) = b.get_type() {
      return b * a;
  }

  if let TensorType::Vector(vector_type) = a.get_type() {
      if b.is_matrix() {
          match vector_type {
              VectorType::Bra => {
                  assert_eq!(a.col_count(), b.col_count());
                  let mut result = Tensor::empty();
                  b.rows().for_each(|row| result.append_row(&row * a));
                  return result;
              }
              VectorType::Ket => {
                  assert_eq!(a.row_count(), b.row_count());
                  let mut result = Tensor::empty();
                  b.cols().for_each(|col| result.append_col(&col * a));
                  return result;
              }
          }
      }
  }
  
  if let TensorType::Vector(_) = b.get_type() {
      if a.is_matrix() {
          return b * a;
      }
  }
  panic!("Unsupported shape {:?} vs {:?}", a.shape, b.shape);
}