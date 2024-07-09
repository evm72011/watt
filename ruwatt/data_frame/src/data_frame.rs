use std::collections::HashMap;

use num::Float;
use tensor::{IndexError, Tensor};
use super::{FrameData, FrameHeader};
#[derive(Debug)]
pub struct DataFrame<T=f64> where T: Float {
    pub data: Vec<FrameData<T>>,
    pub headers: Vec<FrameHeader<T>>
}

impl<T> DataFrame<T> where T: Float + Default {
    pub fn new() -> Self {
        DataFrame {
            data: vec![],
            headers: vec![]
        }
    }

    pub fn col_count(&self) -> usize {
        self.headers.len()
    }

    pub fn row_count(&self) -> usize {
        self.data.len() / self.headers.len()
    }

    pub fn row(&self, index: usize) -> Result<Vec<FrameData<T>>, IndexError> {
        let row_count = self.row_count();
        let col_count = self.col_count();
        if index < row_count {
            let start = col_count * index;
            let end = col_count * (index + 1);
            return Ok(self.data[start..end].to_vec());
        }
        Err(IndexError::IndexOutOfBounds)
    }

    pub fn col(&self, index: usize) -> Result<Vec<FrameData<T>>, IndexError> {
        let row_count = self.row_count();
        let col_count = self.col_count();
        if index < col_count {
            let result = (0..row_count)
                .map(|row| self.data[col_count * row + index].clone())
                .collect();
            return Ok(result);             
        }
        Err(IndexError::IndexOutOfBounds)
    }

    pub fn to_tensor(&self) -> Tensor<T> {
        let all_numbers = self.headers.iter().all(|header| matches!(header.data_type, FrameData::Number(_)));
        assert!(all_numbers, "Must contain numbers only");
        Tensor::<T>::empty()
    }

    fn get_header_index(&self, name: &str) -> usize {
        let name = String::from(name);
        self.headers.iter()
            .position(|item| item.name == name)
            .unwrap_or_else(|| panic!("Column {} not found", name))
    }

    pub fn apply(&mut self, map: HashMap<&str, Box<dyn Fn(&FrameData<T>) -> FrameData<T>>>) {
        for (name, mapper) in map.into_iter() {
            let col_index = self.get_header_index(name);
            let header = &mut self.headers[col_index];
            
            if FrameData::NA != mapper(&header.data_type) {
                header.data_type = mapper(&header.data_type).default();
            }

            let row_count = self.row_count();
            let col_count = self.col_count();
            (0..row_count)
                .map(|row| col_count * row + col_index)
                .for_each(|index| self.data[index] = mapper(&self.data[index]));
        }
    }
}
