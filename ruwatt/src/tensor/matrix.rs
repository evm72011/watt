use num::Float;
use crate::{assert_bra, assert_ket, assert_matrix};

use super::Tensor;

impl<T> Tensor<T> where T: Float {    
    pub fn identity(size: usize) -> Self {
        let mut result = Self::zeros(vec![ size, size ]);
        for i in 0..size {
            result.set(vec![i, i], T::one());
        }
        result
    }

    pub fn matrix(data: Vec<Vec<T>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        Tensor {
            shape: vec![rows, cols],
            data: data.into_iter().flatten().collect()
        }
    }

    pub fn is_matrix(&self) -> bool {
        self.shape.len() == 2
    }

    pub fn is_square_matrix(&self) -> bool {
        self.shape.len() == 2 && self.shape[0] == self.shape[1]
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
        self.shape[0]
    }

    pub fn col_count(&self) -> usize {
        assert_matrix!(self);
        self.shape[1]
    }

    pub fn row(&self, row: usize) -> Self {
        assert_matrix!(self);
        let col_count = self.col_count();
        let start = col_count * row;
        let end = col_count * (row + 1);
        let data: Vec<T> = self.data[start..end].to_vec();
        Tensor::<T>::bra(data)
    }

    pub fn col(&self, col: usize) -> Self {
        assert_matrix!(self);
        let row_count = self.row_count();
        let col_count = self.col_count();
        let data = (0..row_count)
            .map(|row| {
                let index = col_count * row + col;
                self.data[index]
            })
            .collect();
        Tensor::<T>::ket(data)
    }

    pub fn append_row(&mut self, row: Tensor<T>) {
        assert_matrix!(self);
        assert_bra!(row);
        assert_eq!(self.col_count(), row.dim(), "Size mismatch");
        self.shape[0] = self.shape[0] + 1;
        self.data.extend(row.data);
    }

    pub fn append_col(&mut self, col: Tensor<T>) {
        assert_matrix!(self);
        assert_ket!(col);
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
    use super::Tensor;

    fn matrix123() -> Tensor {
        Tensor::matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0]
        ])
    }

    #[test]
    fn identity() {
        let matrix = Tensor::<f32>::identity(2);
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
        let vector = Tensor::bra(vec![1.0, 2.0, 3.0]);
        let expected = Tensor::ket(vec![1.0, 2.0, 3.0]);
        let recieved = vector.tr();
        assert!(recieved == expected);
    }

    #[test]
    fn tr_matrix() {
        let matrix = matrix123();
        let expected = Tensor::matrix(vec![
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
        let expected = Tensor::bra(vec![4.0, 5.0, 6.0]);
        let recieved = matrix.row(1);
        assert_eq!(expected, recieved);
    }

    #[test]
    fn col() {
        let matrix = matrix123();
        let expected = Tensor::ket(vec![2.0, 5.0, 8.0]);
        let recieved = matrix.col(1);
        assert_eq!(expected, recieved);
    }

    #[test]
    fn append_row() {
        let mut matrix = Tensor::matrix(vec![
            vec![1.0, 2.0], 
            vec![3.0, 4.0]
        ]);
        let row = Tensor::bra(vec![5.0, 6.0]);
        matrix.append_row(row);

        let expected = Tensor::matrix(vec![
            vec![1.0, 2.0], 
            vec![3.0, 4.0],
            vec![5.0, 6.0]
        ]);
        assert_eq!(matrix, expected);
    }

        #[test]
    fn append_col() {
        let mut matrix = matrix123();
        let col = Tensor::ket(vec![10.0, 11.0, 12.0]);
        matrix.append_col(col);

        let expected = Tensor::matrix(vec![
            vec![1.0, 2.0, 3.0, 10.0], 
            vec![4.0, 5.0, 6.0, 11.0],
            vec![7.0, 8.0, 9.0, 12.0]
        ]);
        assert_eq!(matrix, expected);
    }
}