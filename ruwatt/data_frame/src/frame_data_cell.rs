use num::Float;

#[derive(Debug, Clone, PartialEq)]
pub enum FrameDataCell<T=f64> where T: Float {
    Number(T),
    String(String),
    NA
}

impl<T> FrameDataCell<T> where T: Float + Default {
    pub fn default(&self) -> FrameDataCell<T> {
        match self {
            FrameDataCell::Number(_) => FrameDataCell::Number(Default::default()),
            FrameDataCell::String(_) => FrameDataCell::String(Default::default()),
            FrameDataCell::NA => FrameDataCell::NA
        }
    }
}
