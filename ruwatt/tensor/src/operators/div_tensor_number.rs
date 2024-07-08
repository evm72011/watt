use std::ops;
use num::Float;
use super::super::Tensor;

fn div_tensor_number<T: Float>(a: &Tensor<T>, b: &T) -> Tensor<T> {
    let data = a.data.iter().map(|&a| a / *b).collect();
    Tensor {
        data,
        shape: a.shape.to_vec(),
    }
}

impl<T> ops::Div<&T> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: &T) -> Tensor<T>  {
        div_tensor_number(self, other)
    }
}

impl<T> ops::Div<&T> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: &T) -> Tensor<T>  {
        div_tensor_number(&self, other)
    }
}

impl<T> ops::Div<T> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: T) -> Tensor<T>  {
        div_tensor_number(self, &other)
    }
}

impl<T> ops::Div<T> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: T) -> Tensor<T>  {
        div_tensor_number(&self, &other)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::Vector;
    #[test]
    fn div_trensor_number() {
        let a = Vector::bra(vec![ 2.0, 4.0 ]);
        let expected = Vector::bra(vec![ 1.0, 2.0 ]);
        assert_eq!(expected, a / 2.0)
    }
}