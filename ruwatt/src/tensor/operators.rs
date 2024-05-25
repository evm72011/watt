use std::ops;
use num::Float;
use super::Tensor;

impl<T> ops::Add<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn add(self, other: &Tensor<T>) -> Tensor<T>  {
        self.compare_shape(other);
        let data = self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect();
        Tensor {
            data,
            shape: self.shape.to_vec()
        }
    }
}

impl<T> ops::Sub<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn sub(self, other: &Tensor<T>) -> Self::Output  {
        self.compare_shape(other);
        let data = self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect();
        Tensor {
            data,
            shape: self.shape.to_vec()
        }
    }
}

impl<T> ops::Mul<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn mul(self, other: &Tensor<T>) -> Self::Output  {
        self.compare_shape(other);
        let data = self.data.iter().zip(&other.data).map(|(&a, &b)| a * b).collect();
        Tensor {
            data,
            shape: self.shape.to_vec()
        }
    }
}

impl<T> ops::Mul<&T> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn mul(self, other: &T) -> Self::Output  {
        let data = self.data.iter().map(|&a| a * *other).collect();
        Tensor {
            data,
            shape: self.shape.to_vec()
        }
    }
}

/*
impl<T> ops::Mul<T> for T where T: Float + fmt::Display  {
    type Output = Tensor<T>;

    fn mul(self, other: &Tensor<T>) -> Tensor<T>  {
        let data = other.data.iter().map(|&a| a * self).collect();
        Tensor {
            data,
            shape: self.shape.to_vec()
        }
    }
}
*/


impl<T> ops::Div<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: &Tensor<T>) -> Tensor<T>  {
        self.compare_shape(other);
        let data = self.data.iter().zip(&other.data).map(|(&a, &b)| a / b).collect();
        Tensor {
            data,
            shape: self.shape.to_vec()
        }
    }
}