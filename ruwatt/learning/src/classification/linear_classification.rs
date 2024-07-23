use num::Float;
use optimization::GradientDescent;
use std::fmt::Debug;
use tensor::Tensor;

pub struct LinearClassification<'a, T=f64> where T: Float + Debug {
    pub trained: bool,
    pub coef: Tensor<T>,
    pub optimizator: GradientDescent<'a, T>
}

impl<'a, T> Default for LinearClassification<'a, T> where T: Float + Debug {
    fn default() -> Self {
        Self {
            trained: false,
            coef: Tensor::empty(),
            optimizator: Default::default()
        }
    }
}

impl<'a, T> LinearClassification<'a, T> where T: Float + Debug {
    pub fn fit(&mut self, _x: &Tensor<T>, _y: &Tensor<T>){

    }

    pub fn predict(&mut self, x: &Tensor<T>) -> Tensor<T> {
        x.clone()
    }
}
