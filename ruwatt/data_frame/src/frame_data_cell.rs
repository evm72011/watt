use num::Float;
use std::str::FromStr;
use std::fmt;
use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub enum FrameDataCell<T=f64> where T: Float {
    Number(T),
    String(String),
    NA
}

#[derive(Debug)]
pub struct ParseFrameDataCellError;

impl std::error::Error for ParseFrameDataCellError {}

impl fmt::Display for ParseFrameDataCellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid enum variant")
    }
}

impl<T> FromStr for FrameDataCell<T> where T: Float {
    type Err = ParseFrameDataCellError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let number_pattern = Regex::new(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$").unwrap();

        let result = if s.starts_with('"') || s.ends_with('"') {
            FrameDataCell::<T>::String(s[1..s.len()-1].to_string())
        } else if number_pattern.is_match(s) {
            let value: f64 = s.parse().unwrap();
            FrameDataCell::<T>::Number(T::from(value).unwrap())
        } else if s.len() == 0 {
            FrameDataCell::<T>::NA
        } else {
            FrameDataCell::String(s.to_string())
        };
        Ok(result)
    }
}

impl<T> FrameDataCell<T> where T: Float + Default {
    pub fn default(&self) -> Self {
        match self {
            Self::Number(_) => Self::Number(Default::default()),
            Self::String(_) => Self::String(Default::default()),
            Self::NA => Self::NA
        }
    }

    pub fn numbers(values: &[T]) -> Vec<Self> {
        values.iter()
            .map(|&value| FrameDataCell::<T>::Number(value))
            .collect()
    }
}
