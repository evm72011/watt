use std::ops;
use num::Float;
use super::super::Tensor;
use crate::assert_shape;

fn add_tensors<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_shape!(a, b);
    let data = a.data.iter().zip(&b.data).map(|(&a, &b)| a + b).collect();
    Tensor {
        data,
        shape: a.shape.to_vec(),
    }
}

impl<T> ops::Add<&Tensor<T>> for &Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn add(self, other: &Tensor<T>) -> Tensor<T>  {
        add_tensors(self, other)
    }
}

impl<T> ops::Add<Tensor<T>> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn add(self, other: Tensor<T>) -> Tensor<T>  {
        add_tensors(&self, &other)
    }
}


impl<T> ops::Add<&Tensor<T>> for Tensor<T> where T: Float {
    type Output = Tensor<T>;

    fn add(self, other: &Tensor<T>) -> Tensor<T>  {
        add_tensors(&self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    #[should_panic(expected = "Shapes do not match: [1, 2] vs [2, 1]")]
    fn add_error() {
        let bra = Tensor::bra(vec![ 1.0, 2.0 ]);
        let ket = Tensor::ket(vec![ 1.0, 2.0 ]);
        let _ = bra + ket;
    }

    #[test]
    fn add() {
        let a = Tensor::bra(vec![ 1.0, 2.0 ]);
        let b = Tensor::bra(vec![ 3.0, 4.0 ]);
        let expected = Tensor::bra(vec![ 4.0, 6.0 ]);
        let recieved = a + b;
        assert_eq!(expected, recieved)
    }
}
