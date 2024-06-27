use num::Float;
use std::marker::PhantomData;
use crate::{assert_bra, assert_ket, assert_matrix, tensor::index_tools::IndexError};
use super::super::{Tensor, IndexTools, Vector};

pub struct Matrix<T = f32> {
    _marker: PhantomData<T>
}

impl<T> Matrix<T> where T: Float {
    pub fn ident(size: usize) -> Tensor<T> {
        let mut result = Tensor::zeros(vec![ size, size ]);
        for i in 0..size {
            result.set(vec![i, i], T::one());
        }
        result
    }

    pub fn new(data: Vec<Vec<T>>) -> Tensor<T> {
        let rows = data.len();
        let cols = data[0].len();
        Tensor {
            shape: vec![rows, cols],
            data: data.into_iter().flatten().collect()
        }
    }

    pub fn concat_h(a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
        assert_eq!(
            a.row_count(), 
            b.row_count(), 
            "Incompatible shape {:?} vs {:?}", a.shape, b.shape
        );
        let mut result = Tensor::<T>::empty();
        result.assign(a);
        b.cols().for_each(|col| result.append_col(col));
        result
    }

    pub fn concat_v(a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
        assert_eq!(
            a.col_count(), 
            b.col_count(), 
            "Incompatible shape {:?} vs {:?}", a.shape, b.shape
        );
        let mut result = Tensor::<T>::empty();
        result.assign(a);
        b.rows().for_each(|row| result.append_row(row));
        result
    }
}

impl<T> Tensor<T> where T: Float {  
    pub fn is_matrix(&self) -> bool {
        self.shape.len() == 2
    }

    pub fn is_square_matrix(&self) -> bool {
        self.is_matrix() && self.row_count() == self.col_count()
    }

    pub fn tr(&self) -> Self {
        assert_matrix!(self);
        let rows = self.row_count();
        let cols = self.col_count();
        let data = if self.is_vector() {
            self.data.to_vec()
        } else {

            let mut result = vec![T::zero(); self.data.len()];
            for i in 0..rows {
                for j in 0..cols {
                    result[j * rows + i] = self.data[i * cols + j].clone();
                }
            }
            result
        };

        Tensor {
            shape: vec![cols, rows],
            data
        }
    }

    pub fn row_count(&self) -> usize {
        assert_matrix!(self);
        IndexTools::<T>::get_row_count(&self.shape)
    }

    pub fn col_count(&self) -> usize {
        assert_matrix!(self);
        IndexTools::<T>::get_col_count(&self.shape)
    }

    pub fn row(&self, index: usize) -> Result<Self, IndexError> {
        assert_matrix!(self);
        let data = IndexTools::<T>::get_row(index, &self.shape, &self.data)?;
        Ok(Vector::<T>::bra(data))
    }

    pub fn col(&self, index: usize) -> Result<Self, IndexError> {
        assert_matrix!(self);
        let data = IndexTools::<T>::get_col(index, &self.shape, &self.data)?;
        Ok(Vector::<T>::ket(data))
    }

    pub fn get_cols(&self, indices: Vec<usize>) -> Result<Self, IndexError> {
        let mut result = Tensor::<T>::empty();
        self.cols().enumerate()
            .filter(|(index, _)| indices.contains(index))
            .for_each(|(_, col)| result.append_col(col));
        Ok(result)
    }

    pub fn append_row(&mut self, row: Tensor<T>) {
        assert_bra!(row);
        if self.is_empty() {
            self.assign(row);
            return;
        }
        assert_matrix!(self);
        assert_eq!(self.col_count(), row.dim(), "Size mismatch");
        self.shape[0] = self.shape[0] + 1;
        self.data.extend(row.data);
    }

    pub fn append_col(&mut self, col: Tensor<T>) {
        assert_ket!(col);
        if self.is_empty() {
            self.assign(col);
            return;
        }

        assert_matrix!(self);
        assert_eq!(self.row_count(), col.dim(), "Size mismatch");
        let col_count = self.col_count();
        self.shape[1] = col_count + 1;
        for (i, value) in col.data.iter().enumerate() {
            self.data.insert(col_count * (i + 1) + i, *value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{ Tensor, Matrix, Vector, TensorType };

    fn matrix123() -> Tensor {
        Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0]
        ])
    }

    #[test]
    fn get_type_matrix() {
        let matrix = Matrix::<f32>::ident(2);
        assert_eq!(matrix.get_type(), TensorType::Matrix);
    }

    #[test]
    fn identity() {
        let matrix = Matrix::<f32>::ident(2);
        assert_eq!(matrix.shape, vec![2, 2]);
        assert_eq!(matrix.data, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn matrix() {
        let matrix = matrix123();
        assert_eq!(matrix.shape, vec![3, 3]);
        let data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        assert_eq!(matrix.data, data);
    }

    #[test]
    fn tr_vector() {
        let vector = Vector::bra(vec![1.0, 2.0, 3.0]);
        let expected = Vector::ket(vec![1.0, 2.0, 3.0]);
        let recieved = vector.tr();
        assert!(recieved == expected);
    }

    #[test]
    fn tr_matrix() {
        let matrix = matrix123();
        let expected = Matrix::new(vec![
            vec![1.0, 4.0, 7.0], 
            vec![2.0, 5.0, 8.0], 
            vec![3.0, 6.0, 9.0]
        ]);
        let recieved = matrix.tr();
        assert_eq!(expected, recieved);
    }

    #[test]
    fn row() {
        let matrix = matrix123();
        let expected = Ok(Vector::bra(vec![4.0, 5.0, 6.0]));
        let recieved = matrix.row(1);
        assert_eq!(expected, recieved);
    }

    #[test]
    fn col() {
        let matrix = matrix123();
        let expected = Ok(Vector::ket(vec![2.0, 5.0, 8.0]));
        let recieved = matrix.col(1);
        assert_eq!(expected, recieved);
    }

    #[test]
    fn append_row() {
        let mut matrix = Matrix::new(vec![
            vec![1.0, 2.0], 
            vec![3.0, 4.0]
        ]);
        let row = Vector::bra(vec![5.0, 6.0]);
        matrix.append_row(row);

        let expected = Matrix::new(vec![
            vec![1.0, 2.0], 
            vec![3.0, 4.0],
            vec![5.0, 6.0]
        ]);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn append_col() {
        let mut matrix = matrix123();
        let col = Vector::ket(vec![10.0, 11.0, 12.0]);
        matrix.append_col(col);

        let expected = Matrix::new(vec![
            vec![1.0, 2.0, 3.0, 10.0], 
            vec![4.0, 5.0, 6.0, 11.0],
            vec![7.0, 8.0, 9.0, 12.0]
        ]);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn concat_h() {
        let a = Vector::ket(vec![1.0, 2.0, 3.0]);
        let b = Vector::ket(vec![4.0, 5.0, 6.0]);
        let recieved = Matrix::concat_h(a, b);

        let expected = Matrix::new(vec![
            vec![1.0, 4.0], 
            vec![2.0, 5.0],
            vec![3.0, 6.0]
        ]);
        assert_eq!(recieved, expected);
    }

    #[test]
    fn concat_v() {
        let a = Vector::bra(vec![1.0, 2.0, 3.0]);
        let b = Vector::bra(vec![4.0, 5.0, 6.0]);
        let recieved = Matrix::concat_v(a, b);

        let expected = Matrix::new(vec![
            vec![1.0, 2.0, 3.0], 
            vec![4.0, 5.0, 6.0]
        ]);
        assert_eq!(recieved, expected);
    }

    #[test]
    fn get_cols() {
        let matrix = matrix123();
        let recieved = matrix.get_cols(vec![0, 2]);
        let expected = Ok(Matrix::new(vec![
            vec![1.0, 3.0],
            vec![4.0, 6.0],
            vec![7.0, 9.0],
        ]));
        assert_eq!(expected, recieved);
    }
}