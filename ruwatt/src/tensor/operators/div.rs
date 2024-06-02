use std::ops;
use num::Float;
use super::super::Tensor;
use crate::assert_shape;

fn div_tensors<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_shape!(a, b);
    let data = a.data.iter().zip(&b.data).map(|(&a, &b)| a / b).collect();
    Tensor {
        data,
        shape: a.shape.to_vec(),
    }
}

fn div_tensor_number<T: Float>(a: &Tensor<T>, b: &T) -> Tensor<T> {
    let data = a.data.iter().map(|&a| a / *b).collect();
    Tensor {
        data,
        shape: a.shape.to_vec(),
    }
}

impl<T> ops::Div<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: &Tensor<T>) -> Tensor<T>  {
        div_tensors(self, other)
    }
}

impl<T> ops::Div<Tensor<T>> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: Tensor<T>) -> Tensor<T>  {
        div_tensors(&self, &other)
    }
}

impl<T> ops::Div<&T> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn div(self, other: &T) -> Tensor<T>  {
        div_tensor_number(self, other)
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
    use super::Tensor;

    #[test]
    #[should_panic(expected = "Shapes do not match: [1, 2] vs [2, 1]")]
    fn div_error() {
        let bra = Tensor::bra(vec![ 1.0, 2.0 ]);
        let ket = Tensor::ket(vec![ 1.0, 2.0 ]);
        let _ = bra / ket;
    }

    #[test]
    fn div_tensors() {
        let a = Tensor::bra(vec![ 4.0, 9.0 ]);
        let b = Tensor::bra(vec![ 2.0, 3.0 ]);
        let expected = Tensor::bra(vec![ 2.0, 3.0 ]);
        let recieved = a / b;
        assert_eq!(expected, recieved)
    }

    
    #[test]
    fn div_tensor_number() {
        let a = Tensor::bra(vec![ 2.0, 4.0 ]);
        let expected = Tensor::bra(vec![ 1.0, 2.0 ]);
        let recieved = a / 2.0;
        assert_eq!(expected, recieved)
    }
}