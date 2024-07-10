use num::Float;
use tensor::Tensor;

use super::super::{DataFrame, FrameDataCell};

impl<T> DataFrame<T> where T: Float {
    pub fn to_tensor(&self, ignored_names: Option<Vec<&str>>) -> Tensor<T> {
        let ignored_names: Vec<String> = ignored_names.unwrap_or(vec![]).iter()
            .map(|&name| String::from(name))
            .collect();

        let headers_are_numbers = self.headers.iter()
            .filter(|&header| !ignored_names.contains(&header.name))
            .all(|header| matches!(header.data_type, FrameDataCell::Number(_)));
        assert!(headers_are_numbers, "Must contain numbers only");

        let ignored_column_indices: Vec<usize> = self.headers.iter().enumerate()
            .filter(|(_, header)| ignored_names.contains(&header.name))
            .map(|(index, _)| index)
            .collect();

        let (row_count, col_count) = self.get_shape();
        let data: Vec<T> = self.data.iter().enumerate()
            .filter(|(index, _)| !ignored_column_indices.contains(&(index % col_count)))
            .map(|(index, value)| {
                if let FrameDataCell::Number(valuee) = value {
                    *valuee
                } else {
                    let row_index = index / col_count;
                    let col_index = index % row_index;
                    panic!("Not a number! row: {row_index}, col: {col_index}")
                }
            })
            .collect();
        Tensor {
            data,
            shape: vec![row_count, col_count - ignored_column_indices.len()]
        }
    }
}