use std::iter::Sum;
use num::Float;
use super::super::{Tensor, Matrix};

impl<T> Tensor<T> where T: Float + Sum {
    pub fn det(&self) -> T {
        assert_square_matrix!(self);
        if self.col_count() == 1 && self.row_count() == 1 {
            self.data[0]
        } else {
            let row_index = 0;
            self.row(row_index).unwrap().data.iter().enumerate()
                .map(|(col_index, &value)| {
                    let sign = Self::cofactor_sign(row_index, col_index);
                    let cofactor = self.minor(row_index, col_index).det();
                    sign * value * cofactor
                })
                .sum()
        }
    }

    pub fn inverse(&self) -> Result<Tensor<T>, Box<&str>> where T: Float + Sum {
        assert_square_matrix!(self);
    
        let det = self.det();
        if det == T::zero() {
            return Err(Box::new("Matrix is singular and cannot be inverted"));
        }
    
        let size = self.row_count();
        let mut data: Vec<T> = Vec::with_capacity(size * size);
    
        for row_index in 0..size {
            for col_index in 0..size {
                let cofactor = Self::cofactor_sign(row_index, col_index) * 
                               self.minor(row_index, col_index).det();
                data.push(cofactor);
            }
        }
    
        let result = Matrix::square(data).tr();
        Ok(result / det)
    }

    fn minor(&self, row_index: usize, col_index: usize) -> Tensor<T> where T: Float {
        let mut data: Vec<T> = Vec::with_capacity((self.row_count() - 1) * (self.col_count() - 1));
    
        (0..self.row_count())
            .filter(|&i| i != row_index)
            .for_each(|i| {
                (0..self.col_count())
                    .filter(|&j| j != col_index)
                    .for_each(
                        |j| data.push(self.data[i * self.col_count() + j])
                    );
            });
    
        Tensor {
            shape: vec![self.row_count() - 1, self.col_count() - 1],
            data
        }
    } 
    
    fn cofactor_sign(row_index: usize, col_index: usize) -> T where T: Float {
        if (row_index + col_index) % 2 == 0 { T::one() } else { -T::one() }
    }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, Matrix};

    fn matrix123() -> Tensor {
        let data = (1..=9).map(|x| x as f64).collect();
        return Matrix::square(data);
    }

    #[test]
    fn determinant() {
        let recieved = matrix123().det();
        assert_eq!(recieved, 0.0)
    }
    
    #[test]
    fn inverse()  {
        let matrix = Matrix::square(vec![1.0, 2.0, 3.0, 4.0]);
        let recieved = matrix.inverse().unwrap();
        let expected = Matrix::square(vec![-2.0, 1.0, 1.5, -0.5]);
        assert_eq!(recieved, expected);
    }
}
