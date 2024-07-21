use num::Float;
use std::fmt;
use std::{collections::HashMap, error::Error, fmt::Debug};
use super::{DataFrame, FrameDataCell};


pub type ApplyClosure<T> = Box<dyn Fn(&FrameDataCell<T>) -> Result<FrameDataCell<T>, ApplyError>>;

#[derive(Debug)]
pub struct ApplyError(pub String);

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Conversion error: {}", self.0)
    }
}

impl<'a> Error for ApplyError {}

impl<T> DataFrame<T> where T: Float + Default {
    pub fn apply(&mut self, map: HashMap<&str, ApplyClosure<T>>) -> Result<(), ApplyError> {
        for (name, mapper) in map.into_iter() {
            let col_index = self.get_col_index(name);
            let header = &mut self.headers[col_index];
            
            if FrameDataCell::NA != mapper(&header.data_type)? {
                header.data_type = mapper(&header.data_type)?.default();
            }

            self.data.iter_mut().enumerate()
                .for_each(|(index, row)| {
                    row[col_index] = mapper(&row[col_index]).unwrap_or_else(|e| {
                        panic!("Error in line {} occurred: {}", index, e);
                    })
                });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::{mock::df_2x2, ApplyClosure, ApplyError, FrameDataCell};

    #[test]
    fn apply() -> Result<(), ApplyError> {
        let mut df = df_2x2();
        
        let mut map: HashMap<&str, ApplyClosure::<f64>> = HashMap::new();
        map.insert("foo", Box::new(&add_two));
    
        df.apply(map)?;

        assert_eq!(df.row(0).unwrap(), FrameDataCell::numbers(&[3.0, 2.0]));
        assert_eq!(df.row(1).unwrap(), FrameDataCell::numbers(&[5.0, 4.0]));

        return Ok(());
        
        fn add_two(value: &FrameDataCell) -> Result<FrameDataCell, ApplyError> {
            if let FrameDataCell::Number(value) = value {
                Ok(FrameDataCell::Number(value + 2.0))
            } else {
                Err(ApplyError("Value in cell is not a string".into()))
            }
        }    
    }
}
