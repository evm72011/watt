use num::Float;
use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub enum FrameDataCell<T=f64> where T: Float {
    Number(T),
    String(String),
    NA
}

impl<T> FrameDataCell<T> where T: Float + Default {
    pub fn default(&self) -> Self {
        match self {
            Self::Number(_) => Self::Number(Default::default()),
            Self::String(_) => Self::String(Default::default()),
            Self::NA => Self::NA
        }
    }

    pub fn from(value: &str) -> Self {
        let number_pattern = Regex::new(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$").unwrap();
        if value.starts_with('"') || value.ends_with('"') {
            FrameDataCell::<T>::String(value[1..value.len()-1].to_string())
        } else if number_pattern.is_match(value) {
            let value: f64 = value.parse().unwrap();
            FrameDataCell::<T>::Number(T::from(value).unwrap())
        } else if value.len() == 0 {
            FrameDataCell::<T>::NA
        } else {
            FrameDataCell::String(value.to_string())
        }
    }
}
