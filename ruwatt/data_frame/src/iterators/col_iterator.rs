use num::Float;
use super::super::{DataFrame, FrameDataCell};

pub struct ColIterator<'a, T> where T: Float + 'a {
    data_frame: &'a DataFrame<T>,
    index: usize
}

impl<'a, T> Iterator for ColIterator<'a, T> where T: Float {
    type Item = Vec<FrameDataCell<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        self.data_frame.col(self.index - 1).ok()
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

#[cfg(test)]
mod tests {
    use crate::{mock::df_2x2, FrameDataCell};

    #[test]
    fn cols() {
        let df = df_2x2();
        let mut iterator = df.cols();
        assert_eq!(iterator.next(), Some(FrameDataCell::numbers(&[1.0, 3.0])));
        assert_eq!(iterator.next(), Some(FrameDataCell::numbers(&[2.0, 4.0])));
        assert_eq!(iterator.next(), None);
    }
}
