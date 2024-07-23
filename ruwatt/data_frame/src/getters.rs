use std::collections::HashMap;

use num::Float;
use super::{DataFrame, FrameDataCell};
use tensor::IndexError;

impl<T> DataFrame<T> where T: Float {
    pub fn shape(&self) -> (usize, usize) {
        (self.row_count(), self.col_count())
    }

    pub fn col_count(&self) -> usize {
        if self.data.len() == 0 { 
            0 
        } else { 
            self.data[0].len()
        }
    }

    pub fn row_count(&self) -> usize {
        self.data.len()
    }

    pub fn row(&self, index: usize) -> Result<Vec<FrameDataCell<T>>, IndexError> {
        if index < self.row_count() {
            Ok(self.data[index].clone())
        } else {
            Err(IndexError::IndexOutOfBounds)
        }
    }

    pub fn col(&self, index: usize) -> Result<Vec<FrameDataCell<T>>, IndexError> {
        if index < self.col_count() {
            let result = self.data.iter().map(|val| val[index].clone()).collect();
            Ok(result)
        } else {
            Err(IndexError::IndexOutOfBounds)
        }
    }

    pub fn get_col_index(&self, name: &str) -> usize {
        let name = String::from(name);
        self.headers.iter()
            .position(|header| header.name == name)
            .unwrap_or_else(|| panic!("Column {} not found", name))
    }

    pub fn filter<F>(&self, predicate: F) -> Self     
    where
        F: Fn(&Vec<FrameDataCell<T>>) -> bool {
        let data = self.rows()
            .filter(|row| predicate(row))
            .collect();
        Self {
            data,
            headers: self.headers.clone()
        }
    }

    //TODO unit test, naming, iterator
    pub fn row_(&self, index: usize) -> Result<HashMap<&str, FrameDataCell<T>>, IndexError> {
        let result: HashMap<&str, FrameDataCell<T>> = self.row(index)?.iter()
            .zip(self.headers.iter())
            .map(|(cell, header)| (header.name.as_str(), cell.clone()))
            .collect();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::{mock::df_2x3, FrameDataCell};

    #[test]
    fn col_count() {
        let df = df_2x3();
        assert_eq!(df.col_count(), 2)
    }

    #[test]
    fn row_count() {
        let df = df_2x3();
        assert_eq!(df.row_count(), 3)
    }

    #[test]
    fn row() {
        let df = df_2x3();
        
        let recieved = df.row(0).unwrap();
        let expected = FrameDataCell::numbers(&[1.0, 2.0]);

        assert_eq!(recieved, expected)
    }

    #[test]
    fn col() {
        let df = df_2x3();
        
        let recieved = df.col(0).unwrap();
        let expected = FrameDataCell::numbers(&[1.0, 3.0, 5.0]);

        assert_eq!(recieved, expected)
    }

    #[test]
    fn get_col_index() {
        let df = df_2x3();
        assert_eq!(df.get_col_index("foo"), 0)
    }

    /*
    #[test]
    fn col_count() {
        let df = df_2x3();
        let recieved: Vec<f64> = df.get("foo");
        assert_eq!(df.col_count(), 2)
    }
    */
}
