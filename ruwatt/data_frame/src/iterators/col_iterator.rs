use num::Float;
use super::super::{DataFrame, FrameDataCell};

pub struct ColIterator<'a, T> where T: Float + 'a {
    data_frame: &'a DataFrame<T>,
    index: usize
}

impl<'a, T> Iterator for ColIterator<'a, T> where T: Float {
    type Item = Vec<FrameDataCell<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.data_frame.col(self.index).ok()
    }
}

impl<T> DataFrame<T> where T: Float {
    pub fn cols(&self) -> ColIterator<T> {
        ColIterator {
            data_frame: self,
            index: 0,
        }
    }
}
