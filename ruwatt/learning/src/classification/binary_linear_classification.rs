use num::Float;
use optimization::GradientDescent;
use std::{fmt::Debug, iter::Sum};
use tensor::{assert_matrix, dot, Tensor, Vector};

use super::sigmoid;

#[derive(PartialEq, Clone)]
pub enum BinaryLinearClassificationCost {
    LeastSquares,
    CrossEntropy,
    Softmax
}

pub struct BinaryLinearClassification<'a, T=f64> where T: Float + Debug {
    pub coef: Tensor<T>,
    pub cost_function: BinaryLinearClassificationCost,
    pub optimizator: GradientDescent<'a, T>
}

impl<'a, T> Default for BinaryLinearClassification<'a, T> where T: Float + Debug {
    fn default() -> Self {
        Self {
            coef: Tensor::empty(),
            cost_function: BinaryLinearClassificationCost::LeastSquares,
            optimizator: Default::default()
        }
    }
}

impl<'a, T> BinaryLinearClassification<'a, T> where T: Float + Debug + Sum {
    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>){
        self.validate_fit(x, y);
        let f = |w: &Tensor<T>| self.create_cost_function(w, &x, &y);
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
            let value = sigmoid(dot(&self.coef, &x_modified).to_scalar()).round();
            data.push(value)
        });
        Vector::ket(data)
    }

    fn create_cost_function(&self, w: &Tensor<T>, x: &Tensor<T>, y: &Tensor<T>) -> T {
        let count = T::from(y.data.len()).unwrap();
        x.rows()
            .zip(y.data.iter())
            .map(|(x_test, &y_test)| {
                let x_modified = x_test.prepend_one().to_ket();
                let value = sigmoid(dot(&w, &x_modified).to_scalar());
                match self.cost_function {
                    BinaryLinearClassificationCost::LeastSquares => T::powi(value - y_test, 2),
                    BinaryLinearClassificationCost::CrossEntropy => -(y_test * T::ln(value) + (T::one() - y_test) * T::ln(T::one() - value)) / count,
                    BinaryLinearClassificationCost::Softmax => unimplemented!()
                }
            })
            .sum()
    }

    fn trained(&self) -> bool {
        !self.coef.is_empty()
    }

    fn validate_fit(&self, x: &Tensor<T>, y: &Tensor<T>) {
        assert_matrix!(x);
        assert_matrix!(y);
        assert_eq!(x.row_count(), y.row_count(), "Count of x train not correspond to y");

        if BinaryLinearClassificationCost::CrossEntropy == self.cost_function && 
           y.data.iter().any(|&x| x != T::zero() && x != T::one()) {
            panic!("CrossEntropy is only applicable for y values of 0 or 1");
    } 

    }
}
