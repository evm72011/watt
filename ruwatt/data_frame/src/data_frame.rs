use std::error::Error;
use super::{FrameData, FrameHeader};

#[derive(Debug)]
pub struct DataFrame {
    pub data: Vec<FrameData>,
    pub headers: Vec<FrameHeader>
}

impl DataFrame {
    pub fn new() -> Self {
        DataFrame {
            data: vec![],
            headers: vec![]
        }
    }

    pub fn col_count(&self) -> usize {
        self.headers.len()
    }

    pub fn row_count(&self) -> usize {
        self.data.len() / self.headers.len()
    }

    pub fn save_csv(&self, _file_name: &str) -> Result<(), Box<dyn Error>> {
        //for i in 0..self.row_count() {
        //}
        Ok(())
    }

}

