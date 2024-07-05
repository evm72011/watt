use std::error::Error;

pub enum DataType {
    Bool(bool),
    Number(f32),
    String(String)
}

pub struct DataColumn {
    pub data_type: DataType,
    pub name: String
}

pub struct DataFrame {
    pub data: Vec<Option<DataType>>,
    pub columns: Vec<DataColumn>
}

impl DataFrame {
    pub fn save_csv(&self, file_name: &str) -> Result<(), Box<dyn Error>> {
        //for i in 0..self.row_count() {

        //}
        Ok(())
    }

    pub fn read_csv(file_name: &str) {

    }
}