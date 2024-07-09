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

    pub fn row(&self, _index: usize) -> Vec<FrameData> {
        vec![]
    }

    pub fn col(&self, _index: usize) -> Vec<FrameData> {
        vec![]
    }
}

