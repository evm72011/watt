use num::Float;
use crate::assert_matrix;

use crate::tensor::{Tensor, dot::dot, abs};


pub struct LinearRegression<T=f32> where T: Float {
    pub coef: Tensor<T>,
    pub bias: Tensor<T>
}

impl<T> Default for LinearRegression<T> where T: Float  {
    fn default() -> Self {
        Self {
            coef: Tensor::zeros(vec![1]),
            bias: Tensor::zeros(vec![1])
        }
    }
}

fn least_squares<T>(x: Tensor<T>, y: Tensor<T>) where T: Float {

}

impl<T> LinearRegression<T> where T: Float {
    pub fn fit(&mut self, x: Tensor<T>, y: Tensor<T>) {
        assert_matrix!(x);
        assert_matrix!(y);
        assert_eq!(x.shape[0], y.shape[0], "Count of x train not correspond to y");
        self.coef = Tensor::<T>::ones(vec![y.shape[1], x.shape[1]]);
        self.bias = Tensor::<T>::ones(vec![y.shape[1], 1]);

        
        let count = x.shape[0];
        let mut error = T::zero();
        for i in 0..count {
            let mut x_i = x.get_row(i);
            x_i.to_ket();
            let mut y_i = y.get_row(i);
            y_i.to_ket();
            let f_i = dot(&self.coef, &x_i) + &self.bias;
            error = error + abs(&(&f_i - &y_i)).length();
            println!("{}", error.to_f32().unwrap());
        }

    }

    pub fn predict(&mut self, x: Tensor<T>) -> Tensor<T> {
        Tensor::zeros(vec![1])
    }
}
