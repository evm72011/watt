use rand::seq::SliceRandom;
use num::{Float, ToPrimitive};
use crate::Tensor;
use crate::assert_matrix;
use rand::{rngs::StdRng, SeedableRng};

impl<T> Tensor<T> where T: Float {
    pub fn split(&self, left_size: f64, seed: u64) -> (Self, Self) {
        assert_matrix!(self);
        let mut indices: Vec<usize> = (0..self.row_count()).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
        let midpoint = (self.row_count().to_f64().unwrap() * left_size).round() as usize;
        let (indices1, _) = indices.split_at(midpoint);
        let mut tensor1 = Tensor::<T>::empty();
        let mut tensor2 = Tensor::<T>::empty();
        for (index, row) in self.rows().enumerate() {
            if indices1.contains(&index) {
                tensor1.append_row(row)
            } else {
                tensor2.append_row(row)
            }
        }
        (tensor1, tensor2)
    }
}  

#[cfg(test)]
mod tests {
    use crate::Matrix;

    #[test]
    fn split() {
        let matrix = Matrix::new(vec![
            vec![ 1.0, 2.0, 3.0 ], 
            vec![ 4.0, 5.0, 6.0 ],
            vec![ 7.0, 8.0, 9.0 ]
        ]);
        let (matrix1, matrix2) = matrix.split(0.33, 1);
        assert_eq!(matrix1.shape, vec![1, 3]);
        assert_eq!(matrix2.shape, vec![2, 3]);
    }
}