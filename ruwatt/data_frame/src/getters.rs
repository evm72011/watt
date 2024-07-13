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
        self.data.len()
    }

    pub fn row(&self, index: usize) -> Result<Vec<FrameDataCell<T>>, IndexError> {
        if self.row_count() < index {
            Ok(self.data[index].clone())
        } else {
            Err(IndexError::IndexOutOfBounds)
        }
    }

    pub fn col(&self, index: usize) -> Result<Vec<FrameDataCell<T>>, IndexError> {
        if self.row_count() < index {
            let result = self.data.iter().map(|val| val[index].clone()).collect();
            Ok(result)
        } else {
            Err(IndexError::IndexOutOfBounds)
        }
    }
}
