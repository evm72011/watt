use super::MatrixData;

pub struct MatrixRowIterator<'a, T> 
where 
    T: Clone + Copy + 'a 
{
    matrix: &'a dyn MatrixData<T>,
    index: usize
}

impl<'a, T> Iterator for MatrixRowIterator<'a, T> where T: Clone + Copy {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let row_count = self.matrix.row_count();
        if self.index < row_count {
            let row = self.matrix.row(self.index).ok()?;
            self.index += 1;
            Some(row)
        } else {
            None
        }
    }
}
