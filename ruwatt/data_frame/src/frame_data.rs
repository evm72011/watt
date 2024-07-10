use num::Float;

#[derive(Debug, Clone, PartialEq)]
pub enum FrameData<T=f64> where T: Float {
    Number(T),
    String(String),
    NA
}

impl<T> FrameData<T> where T: Float + Default {
    pub fn default(&self) -> FrameData<T> {
        match self {
            FrameData::Number(_) => FrameData::Number(Default::default()),
            FrameData::String(_) => FrameData::String(Default::default()),
            FrameData::NA => FrameData::NA
        }
    }
}