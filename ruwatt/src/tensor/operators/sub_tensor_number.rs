use std::ops;
use num::Float;
use super::super::Tensor;

fn sub_tensor_number<T: Float>(a: &Tensor<T>, b: &T) -> Tensor<T> {
    let data = a.data.iter().map(|&a| a - *b).collect();
    Tensor {
        data,
        shape: a.shape.to_vec(),
    }
}

impl<T> ops::Sub<&T> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn sub(self, other: &T) -> Self::Output  {
        sub_tensor_number(self, other)
    }
}

impl<T> ops::Sub<&T> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn sub(self, other: &T) -> Self::Output  {
        sub_tensor_number(&self, other)
    }
}

impl<T> ops::Sub<T> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn sub(self, other: T) -> Self::Output  {
        sub_tensor_number(self, &other)
    }
}

impl<T> ops::Sub<T> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn sub(self, other: T) -> Self::Output  {
        sub_tensor_number(&self, &other)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::Vector;
    #[test]
    fn sub_trensor_number() {
        let a = Vector::bra(vec![ 3.0, 4.0 ]);
        let expected = Vector::bra(vec![ 1.0, 2.0 ]);
        assert_eq!(expected, a - 2.0)
    }
}
