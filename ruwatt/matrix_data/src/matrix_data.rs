use super::IndexError;

pub trait MatrixData<T> where T: Clone + Copy {
    fn row_count(&self) -> usize;
    fn col_count(&self) -> usize;
    fn get_data(&self) -> Vec<T>;

    fn row(&self, index: usize) -> Result<Vec<T>, IndexError> {
        let row_count = self.row_count();
        let col_count = self.col_count();
        let data = self.get_data();

        if index < row_count {
            let start = col_count * index;
            let end = col_count * (index + 1);
            Ok(data[start..end].to_vec())        
        } else {
            Err(IndexError::IndexOutOfBounds)
        }
    }

    fn col(&self, index: usize) -> Result<Vec<T>, IndexError> {
        let row_count = self.row_count();
        let col_count = self.col_count();
        let data = self.get_data();

        if index < col_count {
            let result = (0..row_count)
                .map(|row| data[col_count * row + index])
                .collect();
            Ok(result)             
        } else {
            Err(IndexError::IndexOutOfBounds)
        }

    }
}
