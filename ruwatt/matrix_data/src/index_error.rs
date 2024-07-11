use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq)]
pub enum IndexError {
    IndexOutOfBounds
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexError::IndexOutOfBounds => write!(f, "Index out of bounds"),
        }
    }
}

impl Error for IndexError {}
