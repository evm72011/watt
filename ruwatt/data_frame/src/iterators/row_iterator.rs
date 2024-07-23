use num::{Float, Saturating};
use super::super::{DataFrame, FrameDataCell};

pub struct RowIterator<'a, T> where T: Float + 'a {
    data_frame: &'a DataFrame<T>,
    front_index: usize,
    back_index: usize
}

impl<'a, T> Iterator for RowIterator<'a, T> where T: Float {
    type Item = Vec<FrameDataCell<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.front_index += 1;
        self.data_frame.row(self.front_index - 1).ok()
    }
}

impl<'a, T> DoubleEndedIterator for RowIterator<'a, T>
where
    T: Float + Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_index <= self.back_index {
            let result = self.data_frame.row(self.back_index).ok();
            self.back_index = self.back_index.saturating_sub(1);
            result
        } else {
            None
        }
    }
}

impl<T> DataFrame<T> where T: Float {
    pub fn rows(&self) -> RowIterator<T> {
        RowIterator {
            data_frame: self,
            front_index: 0,
            back_index: self.row_count().saturating_sub(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{mock::df_2x2, FrameDataCell};

    #[test]
    fn rows() {
        let df = df_2x2();
        let mut iterator = df.rows();
        assert_eq!(iterator.next(), Some(FrameDataCell::numbers(&[1.0, 2.0])));
        assert_eq!(iterator.next(), Some(FrameDataCell::numbers(&[3.0, 4.0])));
        assert_eq!(iterator.next(), None);
    }
}
