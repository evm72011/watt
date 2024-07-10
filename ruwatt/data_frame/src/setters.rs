use std::{collections::HashMap, fmt::Debug};
use num::Float;
use super::{DataFrame, FrameDataCell};

impl<T> DataFrame<T> where T: Float + Default + Debug {
    pub fn apply(&mut self, map: HashMap<&str, Box<dyn Fn(&FrameDataCell<T>) -> FrameDataCell<T>>>) {
        for (name, mapper) in map.into_iter() {
            let col_index = self.get_header_index(name);
            let header = &mut self.headers[col_index];
            
            if FrameDataCell::NA != mapper(&header.data_type) {
                header.data_type = mapper(&header.data_type).default(); // + Default
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

    pub fn set_header_type(&mut self, index: usize, value: &FrameDataCell<T>) {
        let value = value.default();
        if FrameDataCell::NA == self.headers[index].data_type {
            self.headers[index].data_type = value;
        } else {
            assert_eq!(self.headers[index].data_type, value);
        }
    }
}
