use num::Float;
use optimization::{GradientDescent, StepSize};
use std::{fmt::Debug, iter::Sum};
use tensor::{assert_matrix, dot, Tensor, Vector};
use super::BinaryLinearClassificationMethod;

pub struct BinaryLinearClassificationModel<'a, T=f64> where T: Float + Debug {
    pub coef: Tensor<T>,
    pub method: BinaryLinearClassificationMethod,
    pub optimizator: GradientDescent<'a, T>
}

impl<'a, T> Default for BinaryLinearClassificationModel<'a, T> where T: Float + Debug {
    fn default() -> Self {
        Self {
            coef: Tensor::empty(),
            method: BinaryLinearClassificationMethod::LeastSquaresSigmoid,
            optimizator: Default::default()
        }
    }
}

impl<'a, T> BinaryLinearClassificationModel<'a, T> where T: Float + Debug + Sum {
    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>){
        self.validate_fit(x, y);
        let f = |w: &Tensor<T>| self.cost_function_wrapper(w, &x, &y);
        let mut optimizator = GradientDescent {
            func: &f,
            start_point: Vector::bra(vec![T::one(); x.col_count() + 1]),
            ..self.optimizator.clone()
        };
        optimizator.run();
        let result = optimizator.result.unwrap();
        self.coef.append_row(result.arg);
    }

    pub fn predict(&mut self, x: &Tensor<T>) -> Tensor<T> {
        assert!(self.trained(), "Model is not trained");
        let mut data = vec![];
        x.rows().for_each(|item| {
            let x_modified = item.prepend_one().to_ket();
            let activation = |v: T| self.method.activation(v);
            let value = activation(dot(&self.coef, &x_modified).to_scalar()).round();
            data.push(value)
        });
        Vector::ket(data)
    }

    fn cost_function_wrapper(&self, w: &Tensor<T>, x: &Tensor<T>, y: &Tensor<T>) -> T {
        let count = T::from(y.data.len()).unwrap();
        x.rows()
            .zip(y.data.iter())
            .map(|(x_test, &y_test)| {
                let x_modified = x_test.prepend_one().to_ket();
                let value = dot(&w, &x_modified).to_scalar();
                let activation = |v: T| self.method.activation(v);
                let _1 = T::one();
                match self.method {
                    BinaryLinearClassificationMethod::LeastSquaresSigmoid | 
                    BinaryLinearClassificationMethod::LeastSquaresTanh => T::powi(activation(value) - y_test, 2),
                    BinaryLinearClassificationMethod::CrossEntropy => 
                        -(y_test * T::ln(activation(value)) + (_1 - y_test) * T::ln(_1 - activation(value))),
                    BinaryLinearClassificationMethod::Softmax => T::ln(_1 + T::exp(-y_test * value))
                }
            })
            .sum::<T>() / count
    }

    fn trained(&self) -> bool {
        !self.coef.is_empty()
    }

    fn validate_fit(&self, x: &Tensor<T>, y: &Tensor<T>) {
        assert_matrix!(x);
        assert_matrix!(y);
        assert_eq!(x.row_count(), y.row_count(), "Count of x train not correspond to y");

        let allowed_values = self.method.allowed_values();
        let condition =  y.data.iter().all(|x| allowed_values.contains(x));
        assert!(condition, "{} is only applicable for y values {:?}", self.method, allowed_values);

        if self.optimizator.step_size == StepSize::Newton &&
           self.optimizator.regularization.is_none() {
            println!("Warning! Newton step size needs regularization here")
        }
    }
}
