use num::Float;
use super::FrameData;

#[derive(Debug)]
pub struct FrameHeader<T> where T: Float {
  pub data_type: FrameData<T>,
  pub name: String
}

impl<T> FrameHeader<T> where T: Float {
    pub fn new(name: String) -> Self {
        FrameHeader::<T> {
            name,
            data_type: FrameData::<T>::NA
        }
    }
}
