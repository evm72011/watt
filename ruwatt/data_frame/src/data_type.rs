#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Bool(bool),
    Float(f64),
    String(String),
    NA
}

impl DataType {
    pub fn default(&self) -> DataType {
        match self {
            DataType::Bool(_) => DataType::Bool(Default::default()),
            DataType::Float(_) => DataType::Float(Default::default()),
            DataType::String(_) => DataType::String(Default::default()),
            DataType::NA => DataType::NA
        }
    }
}
