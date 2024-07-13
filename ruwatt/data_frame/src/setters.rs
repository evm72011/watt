use std::{collections::{HashMap, HashSet}, fmt::Debug};
use num::Float;
use super::{DataFrame, FrameDataCell};

impl<T> DataFrame<T> where T: Float + Default + Debug + Copy {
    pub fn apply(&mut self, map: HashMap<&str, Box<dyn Fn(&FrameDataCell<T>) -> FrameDataCell<T>>>) {
        for (name, mapper) in map.into_iter() {
            let col_index = self.get_header_index(name);
            let header = &mut self.headers[col_index];
            
            if FrameDataCell::NA != mapper(&header.data_type) {
                header.data_type = mapper(&header.data_type).default();
            }

            self.data.iter_mut()
                .for_each(|row| row[col_index] = mapper(&row[col_index]));
        }
    }

    pub fn drop(&mut self, name: &str) {
        let col_index = self.get_header_index(name);
        self.data.remove(col_index);
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
        self.data.extend(df.data);//
    }
    
    pub fn append_cols(&mut self, df: &DataFrame<T>) {
        assert_eq!(self.row_count(), df.row_count());

        let names1: HashSet<_> = self.headers.iter().map(|h| h.name.clone()).collect();
        let names2: HashSet<_> = df.headers.iter().map(|h| h.name.clone()).collect();
        let intersection: Vec<_> = names1.intersection(&names2).collect();
        assert!(intersection.is_empty(), "DataFrames must have unique names");

        self.data.iter_mut().enumerate()
            .for_each(|(index, row)| row.extend(df.data[index].clone()));
        self.headers.extend(df.headers.clone());
    }
}
