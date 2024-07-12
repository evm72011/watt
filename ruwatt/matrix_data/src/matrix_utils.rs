use std::marker::PhantomData;
use super::MatrixError;

pub struct MatrixUtils<T> {
    _marker: PhantomData<T>
}

impl<T> MatrixUtils<T> where T: Copy  {
    pub fn row(data: &Vec<T>, row_count: usize, col_count: usize, index: usize) -> Result<Vec<T>, MatrixError> {
        if index < row_count {
            let start = col_count * index;
            let end = col_count * (index + 1);
            Ok(data[start..end].to_vec())        
        } else {
            Err(MatrixError::IndexOutOfBounds)
        }
    }

    pub fn col(data: &Vec<T>, row_count: usize, col_count: usize, index: usize) -> Result<Vec<T>, MatrixError> {
        if index < col_count {
            let result = (0..row_count)
                .map(|row| data[col_count * row + index])
                .collect();
            Ok(result)             
        } else {
            Err(MatrixError::IndexOutOfBounds)
        }
    }

    pub fn merge_cols(data1: &Vec<T>, col_count1: usize, data2: &Vec<T>, col_count2: usize) -> Result<Vec<T>, MatrixError> {
        let row_count = data1.len() / col_count1;
        let row_count2 = data1.len() / col_count1;
        if row_count != row_count2 {
            return Err(MatrixError::RowsCountNotMatch);
        }
        let mut result = Vec::<T>::with_capacity(data1.len() + data2.len());
        for index in 0..row_count {
            let row1 = Self::row(data1, row_count, col_count1, index)?;
            let row2 = Self::row(data2, row_count, col_count2, index)?;
            result.extend(row1);
            result.extend(row2);
        }
        Ok(result)
    }
}
