use num::Float;
use super::FrameDataCell;

#[derive(Debug, Clone)]
pub struct FrameHeader<T> where T: Float {
  pub data_type: FrameDataCell<T>,
  pub name: String
}

impl<T> PartialEq for FrameHeader<T> where T: Float {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.data_type == other.data_type
    }
}

impl<T> FrameHeader<T> where T: Float {
    pub fn new(name: String) -> Self {
        FrameHeader::<T> {
            name,
            data_type: FrameDataCell::<T>::NA
        }
    }

    pub fn gen_anonym_headers(col_count: usize) -> Vec<FrameHeader<T>>{
        (0..col_count)
            .map(|i| FrameHeader::new(format!("{}", i)))
            .collect()
    }
}
