use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq)]
pub enum DataFrameIOError {
    NotEnoughDataInLine(usize),
    TooMuchDataInLine(usize),
    HeaderParsingError
}
impl fmt::Display for DataFrameIOError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotEnoughDataInLine(line_index) => write!(f, "Not enough data in line {line_index}"),
            Self::TooMuchDataInLine(line_index) => write!(f, "Too much data in line {line_index}"),
            Self::HeaderParsingError => write!(f, "Header parsing error")
        }
    }
}
impl Error for DataFrameIOError {}
