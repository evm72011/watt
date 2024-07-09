#[derive(Debug, Clone, PartialEq)]
pub enum FrameData {
    Number(f64),
    String(String),
    NA
}

impl FrameData {
    pub fn default(&self) -> FrameData {
        match self {
            FrameData::Number(_) => FrameData::Number(Default::default()),
            FrameData::String(_) => FrameData::String(Default::default()),
            FrameData::NA => FrameData::NA
        }
    }
}
