use num::Float;
use crate::assert_matrix;
use crate::Tensor;

pub struct MatrixRowIterator<'a, T> where T: Float + 'a {
    tensor: &'a Tensor<T>,
    index: usize
}

impl<'a, T> Iterator for MatrixRowIterator<'a, T> where T: Float {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let row_count = self.tensor.row_count();
        if self.index < row_count {
            let vector = self.tensor.row(self.index).ok()?;
            self.index += 1;
            Some(vector)
        } else {
            None
        }
    }
}

impl<T> Tensor<T> where T: Float {
    pub fn rows(&self) -> MatrixRowIterator<T> {
        assert_matrix!(self);
        MatrixRowIterator {
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
        let mut iterator = matrix.rows();
        assert_eq!(iterator.next(), Some(Vector::bra(vec![1.0, 0.0])));
        assert_eq!(iterator.next(), Some(Vector::bra(vec![0.0, 1.0])));
        assert_eq!(iterator.next(), None);
    }
}