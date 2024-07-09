#[derive(Debug, Clone, PartialEq)]
pub enum FrameData {
    Bool(bool),
    Float(f64),
    String(String),
    NA
}

impl FrameData {
    pub fn default(&self) -> FrameData {
        match self {
            FrameData::Bool(_) => FrameData::Bool(Default::default()),
            FrameData::Float(_) => FrameData::Float(Default::default()),
            FrameData::String(_) => FrameData::String(Default::default()),
            FrameData::NA => FrameData::NA
        }
    }
}
