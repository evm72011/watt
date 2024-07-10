use num::Float;
use super::{FrameDataCell, FrameHeader};
#[derive(Debug)]
pub struct DataFrame<T=f64> where T: Float {
    pub data: Vec<FrameDataCell<T>>,
    pub headers: Vec<FrameHeader<T>>
}

impl<T> DataFrame<T> where T: Float {
    pub fn new() -> Self {
        DataFrame {
            data: vec![],
            headers: vec![]
        }
    }
}
