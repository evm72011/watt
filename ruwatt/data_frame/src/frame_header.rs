use super::FrameData;

#[derive(Debug)]
pub struct FrameHeader {
  pub data_type: FrameData,
  pub name: String
}

impl FrameHeader {
    pub fn new(name: String) -> Self {
        FrameHeader {
            name,
            data_type: FrameData::NA
        }
    }
}
