use num::Float;

use crate::assert_matrix;

use super::Tensor;

pub struct TensorIterator<T> where T: Float {
    tensor: Tensor<T>,
    index: usize
}

impl<T> Iterator for TensorIterator<T> where T: Float {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let rows = self.tensor.shape[0];
        if self.index < rows {
            let vector = self.tensor.row(self.index);
            self.index += 1;
            Some(vector)
        } else {
            None
        }
    }
}

impl<T> Tensor<T> where T: Float {
    pub fn rows(self) -> TensorIterator<T> {
        assert_matrix!(self);
        TensorIterator {
            tensor: self,
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn rows() {
        let matrix = Tensor::<f32>::identity(2);
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