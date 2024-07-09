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

    pub fn row(&self, index: usize) -> Result<Vec<FrameData>, Box<dyn Error>> {
        let row_count = self.row_count();
        let col_count = self.col_count();
        if index < row_count {
            let start = col_count * index;
            let end = col_count * (index + 1);
            return Ok(self.data[start..end].to_vec());
        }
        Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Index out of bounds")))
    }

    pub fn col(&self, _index: usize) -> Vec<FrameData> {
        vec![]
    }
}

