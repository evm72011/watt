use num::Float;
use std::sync::{Arc, Mutex};

use crate::{assert_matrix, optimization::GradientDescent};
use crate::tensor::{Tensor, dot::dot};

#[derive(PartialEq, Clone)]
pub enum CostFunction {
    LeastSquares,
    Abs,
}

pub struct LinearRegression<'a, T=f32> where T: Float {
    pub coef: Tensor<T>,
    pub cost_function: CostFunction,
    pub optimizator: GradientDescent<'a, T>
}

impl<'a, T> Default for LinearRegression<'a, T> where T: Float  {
    fn default() -> Self {
        Self {
            coef: Tensor::zeros(vec![1]),
            cost_function: CostFunction::LeastSquares,
            optimizator: Default::default()
        }
    }
}

impl<'a, T> LinearRegression<'a, T> where T: Float + Send + Sync + 'static {

    pub fn fit(&mut self, x: Tensor<T>, y: Tensor<T>) {
        assert_matrix!(x);
        assert_matrix!(y);
        assert_eq!(x.row_count(), y.row_count(), "Count of x train not correspond to y");

        let closures = self.create_closures(y.col_count());
        let self_arc = Arc::new(Mutex::new(self));
        
        for (index, closure) in closures.iter().enumerate() {
            let self_arc = Arc::clone(&self_arc);
            let mut self_locked = self_arc.lock().unwrap();
            let x_clone = x.clone();
            let y_clone = y.clone();
            let f = move |w: &Tensor<T>| closure(w, &x_clone, &y_clone);
            let mut optimizator = GradientDescent {
                func: &f,
                start_point: Tensor::bra(vec![T::one(); x.shape[1] + 1]),
                ..self_locked.optimizator.clone()
            };
            optimizator.run();
            let result = optimizator.result.unwrap();
            if index == 0 {
                self_locked.coef = result.arg;
            } else {
                self_locked.coef.append_row(result.arg)
            }
        }
    }



    pub fn predict(&mut self, x: Tensor<T>) -> Tensor<T> {
        let x_modified = x.add_one().to_ket();
        dot(&self.coef, &x_modified)
    }

    fn create_closures(&self, count: usize) -> Vec<Box<dyn Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> T + Send + Sync>>{
        let cost_function = self.cost_function.clone();
        (0..count)
                .map(|index| {
                    let cost_function = cost_function.clone();
                    Box::new(move |w: &Tensor<T>, x: &Tensor<T>, y: &Tensor<T>| {
                        let mut result = T::zero();
                        x.clone().rows()
                            .zip(y.clone().rows())
                            .map(|(x_test, y_test)| {
                                let x_modified = x_test.add_one().to_ket();
                                let value = dot(&w, &x_modified).to_scalar() - *y_test.get_v(index);
                                if cost_function == CostFunction::Abs { 
                                    T::abs(value)
                                } else {
                                    T::powi(value, 2) 
                                }
                            })
                            .for_each(|val| {
                                result = result + val;
                            });
                        result
                    }) as Box<dyn Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> T+ Send + Sync>
            })
            .collect()
    }
}
