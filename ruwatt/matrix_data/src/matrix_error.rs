use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq)]
pub enum MatrixError {
    IndexOutOfBounds,
    RowsCountNotMatch,
    ColumnsCountNotMatch
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::IndexOutOfBounds => write!(f, "Index out of bounds"),
            MatrixError::RowsCountNotMatch=> write!(f, "Rows count don't match"),
            MatrixError::ColumnsCountNotMatch=> write!(f, "Columns count don't match"),
        }
    }
}

impl Error for MatrixError {}
