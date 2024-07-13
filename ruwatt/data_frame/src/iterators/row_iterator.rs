use num::Float;
use super::super::{DataFrame, FrameDataCell};

pub struct RowIterator<'a, T> where T: Float + 'a {
    data_frame: &'a DataFrame<T>,
    index: usize
}

impl<'a, T> Iterator for RowIterator<'a, T> where T: Float {
    type Item = Vec<FrameDataCell<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.data_frame.row(self.index).ok()
    }
}

impl<T> DataFrame<T> where T: Float {
    pub fn rows(&self) -> RowIterator<T> {
        RowIterator {
            data_frame: self,
            index: 0,
        }
    }
}
