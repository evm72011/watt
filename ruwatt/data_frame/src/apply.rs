use num::Float;
use std::fmt;
use std::{collections::HashMap, error::Error, fmt::Debug};
use crate::FrameHeader;

use super::{DataFrame, FrameDataCell};

pub type ApplyClosure<T> = Box<dyn Fn(&FrameDataCell<T>) -> Result<FrameDataCell<T>, ApplyError>>;

pub struct ApplyChanger<T> where T: Float {
    pub cell_changer: ApplyClosure<T>,
    pub new_header: Option<FrameHeader<T>>
}

impl<T> ApplyChanger<T> where T: Float {
    pub fn new(cell_changer: ApplyClosure<T>) -> Self {
        Self {
            cell_changer,
            new_header: None
        }
    }
}

#[derive(Debug)]
pub struct ApplyError(pub String);

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Conversion error: {}", self.0)
    }
}

impl<'a> Error for ApplyError {}

impl<T> DataFrame<T> where T: Float + Default {
    pub fn apply(&mut self, changers: HashMap<&str, ApplyChanger<T>>) -> Result<(), ApplyError> {
        for (name, changer) in changers.into_iter() {
            let col_index = self.get_col_index(name);
            if let Some(header) = changer.new_header {
                self.headers[col_index] = header.clone();
            }

            self.data.iter_mut().enumerate()
                .for_each(|(index, row)| {
                    println!("{index}");
                    row[col_index] = (*changer.cell_changer)(&row[col_index]).unwrap_or_else(|e| {
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
    use crate::{mock::df_2x2, ApplyChanger, ApplyError, FrameDataCell};

    #[test]
    fn apply() -> Result<(), ApplyError> {
        let mut df = df_2x2();
        
        let mut map: HashMap<_, _> = HashMap::new();
        map.insert("foo", ApplyChanger::new(Box::new(&add_two)));
    
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
