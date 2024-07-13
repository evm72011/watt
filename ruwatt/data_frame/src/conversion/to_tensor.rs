use std::{collections::HashSet, fmt::Debug};
use num::Float;
use tensor::Tensor;

use super::super::{DataFrame, FrameDataCell};

impl<T> DataFrame<T> where T: Float + Default + Debug {
    pub fn to_tensor(&self, ignored_names: Option<Vec<&str>>) -> Tensor<T> {
        let ignored_names: HashSet<String> = ignored_names.unwrap_or(vec![]).iter()
            .map(|&name| String::from(name)).collect();

        let col_indices: Vec<usize> = self.headers.iter().enumerate()
            .filter(|(_, header)| !ignored_names.contains(&header.name))
            .map(|(index, _)| index)
            .collect();

        self.headers.iter()
            .filter(|header| !ignored_names.contains(&header.name))
            .for_each(|header| assert_eq!(header.data_type, FrameDataCell::Number(Default::default())));

        let data: Vec<T> = self.rows()
            .map(|row| row.iter().enumerate()
                .filter(|(index, _)| col_indices.contains(index))
                .map(|(_, value)| value.clone())
                .collect::<Vec<FrameDataCell<T>>>()
            )
            .enumerate()
            .flat_map(|(row_index, row)| 
                row.iter().enumerate().map(|(col_index, value)| 
                    if let FrameDataCell::Number(value) = value {
                        *value
                    } else {
                        panic!("Not a number! row: {row_index}, col: {col_index}")
                    }
                ).collect::<Vec<T>>()
            )
            .collect();

        Tensor {
            data,
            shape: vec![self.row_count(), col_indices.len()]
        }
    }
}

#[cfg(test)]
mod tests {
    use tensor::Matrix;
    use crate::mock::df_2x2;

    #[test]
    fn to_tensor() {
        let expected = Matrix::square(vec![1.0, 2.0, 3.0, 4.0]);
        let df = df_2x2();
        let recieved = df.to_tensor(None);
        assert_eq!(expected, recieved);
    }
}
