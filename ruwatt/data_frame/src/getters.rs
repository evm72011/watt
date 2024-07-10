use num::Float;
use super::{DataFrame, FrameDataCell};
use tensor::IndexError;

impl<T> DataFrame<T> where T: Float {
    pub fn get_shape(&self) -> (usize, usize) {
        (self.row_count(), self.col_count())
    }

    pub  fn get_header_index(&self, name: &str) -> usize {
        let name = String::from(name);
        self.headers.iter()
            .position(|header| header.name == name)
            .unwrap_or_else(|| panic!("Column {} not found", name))
    }

    pub fn col_count(&self) -> usize {
        self.headers.len()
    }

    pub fn row_count(&self) -> usize {
        self.data.len() / self.headers.len()
    }

    pub fn row(&self, index: usize) -> Result<Vec<FrameDataCell<T>>, IndexError> {
        let (row_count, col_count) = self.get_shape();
        if index < row_count {
            let start = col_count * index;
            let end = col_count * (index + 1);
            return Ok(self.data[start..end].to_vec());
        }
        Err(IndexError::IndexOutOfBounds)
    }

    pub fn col(&self, index: usize) -> Result<Vec<FrameDataCell<T>>, IndexError> {
        let (row_count, col_count) = self.get_shape();
        if index < col_count {
            let result = (0..row_count)
                .map(|row| self.data[col_count * row + index].clone())
                .collect();
            return Ok(result);             
        }
        Err(IndexError::IndexOutOfBounds)
    }
}
