use super::MatrixData;

pub struct MatrixColIterator<'a, T> 
where 
    T: Clone + Copy + 'a 
{
    matrix: &'a dyn MatrixData<T>,
    index: usize
}

impl<'a, T> Iterator for MatrixColIterator<'a, T> where T: Clone + Copy {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let col_count = self.matrix.col_count();
        if self.index < col_count {
            let column = self.matrix.col(self.index).ok()?;
            self.index += 1;
            Some(column)
        } else {
            None
        }
    }
}
