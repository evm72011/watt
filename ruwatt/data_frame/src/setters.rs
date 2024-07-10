use std::{collections::{HashMap, HashSet}, fmt::Debug};
use num::Float;
use super::{DataFrame, FrameDataCell};

impl<T> DataFrame<T> where T: Float + Default + Debug {
    pub fn apply(&mut self, map: HashMap<&str, Box<dyn Fn(&FrameDataCell<T>) -> FrameDataCell<T>>>) {
        for (name, mapper) in map.into_iter() {
            let col_index = self.get_header_index(name);
            let header = &mut self.headers[col_index];
            
            if FrameDataCell::NA != mapper(&header.data_type) {
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

    pub fn set_header_type(&mut self, index: usize, value: &FrameDataCell<T>) {
        let value = value.default();
        if FrameDataCell::NA == self.headers[index].data_type {
            self.headers[index].data_type = value;
        } else {
            assert_eq!(self.headers[index].data_type, value);
        }
    }

    pub fn rename(&mut self, map: HashMap<&str, &str>) {
        for (name, new_name) in map.into_iter() {
            let col_index = self.get_header_index(name);
            let header = &mut self.headers[col_index];
            header.name = String::from(new_name);
        }
    }

    pub fn append_rows(&mut self, df: DataFrame<T>) {
        assert_eq!(self.col_count(), df.col_count());
        self.data.extend(df.data);
    }
    
    pub fn append_cols(&mut self, df: DataFrame<T>) {
        assert_eq!(self.row_count(), df.row_count());

        let names1: HashSet<_> = self.headers.iter().map(|h| h.name.clone()).collect();
        let names2: HashSet<_> = df.headers.iter().map(|h| h.name.clone()).collect();
        let intersection: Vec<_> = names1.intersection(&names2).collect();
        assert!(intersection.is_empty(), "DataFrames must have unique names");

        let (row_count, col_count) = self.get_shape();
        for row in 0..row_count {
            for col in 0..df.col_count() {
                let value = df.data[row * df.col_count() + col].clone();
                self.data.insert(col_count * (row + 1) + row, value);   //Not checkec
            }
        }
        self.headers.extend(df.headers);
    }
}
