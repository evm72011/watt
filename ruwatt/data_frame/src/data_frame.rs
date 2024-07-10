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
        let (row_count, col_count) = self.get_shape();
        if index < row_count {
            let start = col_count * index;
            let end = col_count * (index + 1);
            return Ok(self.data[start..end].to_vec());
        }
        Err(IndexError::IndexOutOfBounds)
    }

    pub fn col(&self, index: usize) -> Result<Vec<FrameData<T>>, IndexError> {
        let (row_count, col_count) = self.get_shape();
        if index < col_count {
            let result = (0..row_count)
                .map(|row| self.data[col_count * row + index].clone())
                .collect();
            return Ok(result);             
        }
        Err(IndexError::IndexOutOfBounds)
    }

    pub fn to_tensor(&self, ignored_names: Option<Vec<&str>>) -> Tensor<T> {
        let ignored_names: Vec<String> = ignored_names.unwrap_or(vec![]).iter()
            .map(|&name| String::from(name))
            .collect();

        let headers_are_numbers = self.headers.iter()
            .filter(|&header| !ignored_names.contains(&header.name))
            .all(|header| matches!(header.data_type, FrameData::Number(_)));
        assert!(headers_are_numbers, "Must contain numbers only");

        let ignored_column_indices: Vec<usize> = self.headers.iter().enumerate()
            .filter(|(_, header)| ignored_names.contains(&header.name))
            .map(|(index, _)| index)
            .collect();

        let (row_count, col_count) = self.get_shape();
        let data: Vec<T> = self.data.iter().enumerate()
            .filter(|(index, _)| !ignored_column_indices.contains(&(index % col_count)))
            .map(|(_, value)| {
                if let FrameData::Number(valuee) = value {
                    *valuee
                } else {
                    panic!("Not a number")
                }
            })
            .collect();
        Tensor {
            data,
            shape: vec![row_count, col_count - ignored_column_indices.len()]
        }
    }

    fn get_shape(&self) -> (usize, usize) {
        (self.row_count(), self.col_count())
    }

    fn get_header_index(&self, name: &str) -> usize {
        let name = String::from(name);
        self.headers.iter()
            .position(|header| header.name == name)
            .unwrap_or_else(|| panic!("Column {} not found", name))
    }

    pub fn apply(&mut self, map: HashMap<&str, Box<dyn Fn(&FrameData<T>) -> FrameData<T>>>) {
        for (name, mapper) in map.into_iter() {
            let col_index = self.get_header_index(name);
            let header = &mut self.headers[col_index];
            
            if FrameData::NA != mapper(&header.data_type) {
                header.data_type = mapper(&header.data_type).default();
            }

            let (row_count, col_count) = self.get_shape();
            (0..row_count)
                .map(|row| col_count * row + col_index)
                .for_each(|index| self.data[index] = mapper(&self.data[index]));
        }
    }

    pub fn drop(&mut self, name: &str) {
        let col_index = self.get_header_index(name);
        let (row_count, col_count) = self.get_shape();

        let indices: Vec<usize> = (0..row_count)
            .map(|row| col_count * row + col_index)
            .collect();

        self.data = self.data.iter().enumerate()
            .filter(|(index, _)| !indices.contains(index))
            .map(|(_, value)| value.clone())
            .collect();

        self.headers.remove(col_index);
    }
}
