use std::error::Error;
use super::DataType;

pub struct DataFrame {
    pub data: Vec<DataType>,
    pub header_types: Vec<DataType>,
    pub header_names: Vec<String>
}

impl DataFrame {
    pub fn new() -> Self {
        DataFrame {
            data: vec![],
            header_names: vec![],
            header_types: vec![]
        }
    }

    pub fn col_count(&self) -> usize {
        self.header_names.len()
    }

    pub fn row_count(&self) -> usize {
        self.data.len() / self.header_names.len()
    }

    pub fn save_csv(&self, _file_name: &str) -> Result<(), Box<dyn Error>> {
        //for i in 0..self.row_count() {
        //}
        Ok(())
    }

}

