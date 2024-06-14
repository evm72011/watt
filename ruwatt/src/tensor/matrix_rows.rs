use num::Float;

use crate::assert_matrix;

use super::Tensor;

pub struct TensorRowIterator<T> where T: Float {
    tensor: Tensor<T>,
    index: usize
}

impl<T> Iterator for TensorRowIterator<T> where T: Float {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let row_count = self.tensor.row_count();
        if self.index < row_count {
            let vector = self.tensor.row(self.index);
            self.index += 1;
            Some(vector)
        } else {
            None
        }
    }
}

impl<T> Tensor<T> where T: Float {
    pub fn rows(self) -> TensorRowIterator<T> {
        assert_matrix!(self);
        TensorRowIterator {
            tensor: self,
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    //use crate::tensor::Matrix;

    use super::{Tensor, super::Matrix};

    #[test]
    fn rows() {
        let matrix = Matrix::<f32>::ident(2);
        for (index, item) in matrix.clone().rows().enumerate() {
            if index == 0 {
                assert_eq!(item, Tensor::bra(vec![1.0, 0.0]));
            }
            if index == 1 {
                assert_eq!(item, Tensor::bra(vec![0.0, 1.0]));
            }
        }
        let count = matrix.rows().count();
        assert_eq!(count, 2);
    }
}