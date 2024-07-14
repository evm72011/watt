use std::{collections::{HashMap, HashSet}, fmt::Debug};
use num::Float;
use super::{DataFrame, FrameDataCell};

impl<T> DataFrame<T> where T: Float + Default + Debug + Copy {
    pub fn apply(&mut self, map: HashMap<&str, Box<dyn Fn(&FrameDataCell<T>) -> FrameDataCell<T>>>) {
        for (name, mapper) in map.into_iter() {
            let col_index = self.get_col_index(name);
            let header = &mut self.headers[col_index];
            
            if FrameDataCell::NA != mapper(&header.data_type) {
                header.data_type = mapper(&header.data_type).default();
            }

            self.data.iter_mut()
                .for_each(|row| row[col_index] = mapper(&row[col_index]));
        }
    }

    pub fn drop(&mut self, name: &str) {
        let col_index = self.get_col_index(name);
        self.headers.remove(col_index);
        self.data.iter_mut()
            .for_each(|row| {
                row.remove(col_index);
            });
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
            let col_index = self.get_col_index(name);
            let header = &mut self.headers[col_index];
            header.name = String::from(new_name);
        }
    }

    pub fn append_rows(&mut self, df: &DataFrame<T>) {
        assert_eq!(self.col_count(), df.col_count());
        self.data.extend(df.data.clone());
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use tensor::{Matrix, Vector};

    use crate::{mock::df_2x2, FrameDataCell};

    #[test]
    fn apply() {
        let mut df = df_2x2();
        
        let mut map: HashMap<&str, Box<dyn Fn(&FrameDataCell) -> FrameDataCell>> = HashMap::new();
        map.insert("foo", Box::new(&add_two));
    
        df.apply(map);

        assert_eq!(df.row(0).unwrap(), FrameDataCell::numbers(&[3.0, 2.0]));
        assert_eq!(df.row(1).unwrap(), FrameDataCell::numbers(&[5.0, 4.0]));

        fn add_two(value: &FrameDataCell) -> FrameDataCell {
            if let FrameDataCell::Number(value) = value {
                FrameDataCell::Number(value + 2.0)
            } else {
                panic!("Value in cell is not a string")
            }
        }
    }

    #[test]
    fn drop() {
        let mut df = df_2x2();
        df.drop("foo");

        let recieved = df.to_tensor(None);
        let expected = Vector::ket(vec![2.0, 4.0]);
        assert_eq!(recieved, expected)
    }

    #[test]
    fn rename() {
        let mut df = df_2x2();
        let mut map = HashMap::new();
        map.insert("bar", "baz");
        df.rename(map);

        let recieved: Vec<&str> = df.headers.iter().map(|h| h.name.as_str()).collect();
        
        assert_eq!(recieved, vec!["foo", "baz"])
    }

    #[test]
    fn append_rows() {
        let mut df = df_2x2();
        let df_2 = df_2x2();
        df.append_rows(&df_2);

        let recieved = df.to_tensor(None);
        let expected = Matrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);

        assert_eq!(recieved, expected)
    }

    #[test]
    fn append_cols() {
        let mut df = df_2x2();
        let mut map = HashMap::new();
        map.insert("foo", "foo_1");
        map.insert("bar", "bar_1");
        df.rename(map);

        let df_2 = df_2x2();

        df.append_cols(&df_2);

        let recieved = df.to_tensor(None);
        let expected = Matrix::new(vec![
            vec![1.0, 2.0, 1.0, 2.0],
            vec![3.0, 4.0, 3.0, 4.0],
        ]);

        assert_eq!(recieved, expected)
    }

}