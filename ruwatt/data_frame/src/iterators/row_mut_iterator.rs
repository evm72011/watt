use num::Float;
use super::super::{DataFrame, FrameDataCell};

pub struct RowMutIterator<'a, T> where T: Float + 'a {
    data_frame: &'a mut DataFrame<T>,
    index: usize,
}

impl<'a, T> Iterator for RowMutIterator<'a, T> where T: Float {
    type Item = &'a mut Vec<FrameDataCell<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data_frame.data.len() {
            None
        } else {
            let row = self.data_frame.data.get_mut(self.index);
            self.index += 1;
            row
        }
    }
}

impl<T> DataFrame<T> where T: Float {
    pub fn rows_mut(&mut self) -> RowMutIterator<T> {
        RowMutIterator {
            data_frame: self,
            index: 0,
        }
    }
}
