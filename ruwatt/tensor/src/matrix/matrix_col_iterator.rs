use num::Float;
use crate::assert_matrix;
use crate::Tensor;

pub struct MatrixColIterator<'a, T> where T: Float + 'a {
    tensor: &'a Tensor<T>,
    index: usize
}

impl<'a, T> Iterator for MatrixColIterator<'a, T> where T: Float {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let col_count = self.tensor.col_count();
        if self.index < col_count {
            let vector = self.tensor.col(self.index).ok()?;
            self.index += 1;
            Some(vector)
        } else {
            None
        }
    }
}

impl<T> Tensor<T> where T: Float {
    pub fn cols(&self) -> MatrixColIterator<T> {
        assert_matrix!(self);
        MatrixColIterator {
            tensor: self,
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Matrix, Vector};

    #[test]
    fn rows() {
        let matrix = Matrix::<f64>::ident(2);
        let mut iterator = matrix.cols();
        assert_eq!(iterator.next(), Some(Vector::ket(vec![1.0, 0.0])));
        assert_eq!(iterator.next(), Some(Vector::ket(vec![0.0, 1.0])));
        assert_eq!(iterator.next(), None);
    }
}