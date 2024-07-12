use super::{MatrixError, MatrixUtils};

pub trait MatrixData<T> where T: Clone + Copy {
    fn row_count(&self) -> usize;
    fn col_count(&self) -> usize;
    fn get_data(&self) -> Vec<T>;
    fn assign(&mut self, matrix: &Self);

    fn row(&self, index: usize) -> Result<Vec<T>, MatrixError> {
        let data = self.get_data();
        let row_count = self.row_count();
        let col_count = self.col_count();
        MatrixUtils::<T>::row(&data, row_count, col_count, index)
    }

    fn col(&self, index: usize) -> Result<Vec<T>, MatrixError> {
        let data = self.get_data();
        let row_count = self.row_count();
        let col_count = self.col_count();
        MatrixUtils::<T>::col(&data, row_count, col_count, index)
    }

    fn is_empty(&self) -> bool {
        self.get_data().is_empty()
    }
}
