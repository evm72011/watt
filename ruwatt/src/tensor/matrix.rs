use num::Float;
use super::Tensor;
use std::fmt::Debug;

impl<T> Tensor<T> where T: Float + Debug {    
    pub fn identity(size: usize) -> Self {
        let mut result = Self::zeros(vec![ size, size ]);
        for i in 0..size {
            result.set(vec![i, i], T::one());
        }
        result
    }

    pub fn matrix(data: Vec<Vec<T>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        let mut result = Tensor::<T>::zeros(vec![rows, cols]);
        for row in 0..rows {
        for col in 0..cols {
            result.set(vec![row, col], data[row][col]);
        }
        }
        result
    }

    pub fn tr(&self) -> Self {
        if self.shape.len() > 2 {
            unimplemented!("This method is not yet implemented for dim > 2");
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let data = if self.is_vector() {
            self.data.to_vec()
        } else {
            let mut transposed_data = vec![T::zero(); self.data.len()];
            for i in 0..rows {
                for j in 0..cols {
                    transposed_data[j * rows + i] = self.data[i * cols + j].clone();
                }
            }
            transposed_data
        };

        Tensor {
            shape: vec![cols, rows],
            data
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn identity() {
        let matrix = Tensor::<f32>::identity(2);
        assert_eq!(matrix.shape, vec![2, 2]);
        assert_eq!(matrix.data, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn matrix() {
        let matrix = Tensor::matrix(vec![
            vec![1.0, 2.0], 
            vec![3.0, 4.0]
        ]);
        assert_eq!(matrix.shape, vec![2, 2]);
        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn tr_vector() {
        let vector = Tensor::bra(vec![1.0, 2.0, 3.0]);
        let expected = Tensor::ket(vec![1.0, 2.0, 3.0]);
        let recieved = vector.tr();
        assert!(recieved == expected);
    }

    #[test]
    fn tr_matrix() {
        let matrix = Tensor::matrix(vec![
            vec![1.0, 2.0, 3.0], 
            vec![4.0, 5.0, 6.0]
        ]);

        let expected = Tensor::matrix(vec![
            vec![1.0, 4.0], 
            vec![2.0, 5.0], 
            vec![3.0, 6.0]
        ]);
        let recieved = matrix.tr();
        assert_eq!(expected, recieved);
    }
}