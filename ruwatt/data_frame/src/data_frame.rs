use std::error::Error;
use super::DataType;

pub struct DataFrame {
    pub data: Vec<DataType>,
    pub header_types: Vec<DataType>,
    pub header_names: Vec<String>
}

impl DataFrame {
    pub fn save_csv(&self, _file_name: &str) -> Result<(), Box<dyn Error>> {
        //for i in 0..self.row_count() {
        //}
        Ok(())
    }

}

