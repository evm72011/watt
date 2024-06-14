use num::Float;

use crate::assert_matrix;

use super::Tensor;

pub struct TensorColIterator<T> where T: Float {
    tensor: Tensor<T>,
    index: usize
}

impl<T> Iterator for TensorColIterator<T> where T: Float {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let col_count = self.tensor.col_count();
        if self.index < col_count {
            let vector = self.tensor.col(self.index);
            self.index += 1;
            Some(vector)
        } else {
            None
        }
    }
}

impl<T> Tensor<T> where T: Float {
    pub fn cols(self) -> TensorColIterator<T> {
        assert_matrix!(self);
        TensorColIterator {
            tensor: self,
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::matrix::Matrix;

    use super::Tensor;

    #[test]
    fn rows() {
        let matrix = Matrix::<f32>::ident(2);
        for (index, item) in matrix.clone().cols().enumerate() {
            if index == 0 {
                assert_eq!(item, Tensor::ket(vec![1.0, 0.0]));
            }
            if index == 1 {
                assert_eq!(item, Tensor::ket(vec![0.0, 1.0]));
            }
        }
        let count = matrix.rows().count();
        assert_eq!(count, 2);
    }
}