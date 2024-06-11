use num::Float;
use std::iter::Sum;
use std::sync::{Arc, Mutex};

use crate::{assert_matrix, optimization::GradientDescent};
use crate::tensor::{ Tensor, dot };

#[derive(PartialEq, Clone)]
pub enum CostFunction {
    LeastSquares,
    Abs,
}

pub struct LinearRegression<'a, T=f32> where T: Float + Sum {
    pub coef: Tensor<T>,
    pub cost_function: CostFunction,
    pub optimizator: GradientDescent<'a, T>
}

impl<'a, T> Default for LinearRegression<'a, T> where T: Float + Sum {
    fn default() -> Self {
        Self {
            coef: Tensor::zeros(vec![1]),
            cost_function: CostFunction::LeastSquares,
            optimizator: Default::default()
        }
    }
}

impl<'a, T> LinearRegression<'a, T> where T: Float + Send + Sync + Sum + 'static {

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
        let mut result = Tensor::zeros(vec![]);
        x.rows().enumerate().for_each(|(index, item)| {
            let x_modified = item.prepend_one().to_ket();
            let row = dot(&self.coef, &x_modified).to_bra();
            if index == 0 {
                result = row;
            } else {
                result.append_row(row)
            }
        });
        result
        //let x_modified = x.prepend_one();
        //dot(&self.coef, &x_modified)
    }

    fn create_closures(&self, count: usize) -> Vec<Box<dyn Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> T + Send + Sync>>{
        let cost_function = self.cost_function.clone();
        (0..count)
                .map(|index| {
                    let cost_function = cost_function.clone();
                    Box::new(move |w: &Tensor<T>, x: &Tensor<T>, y: &Tensor<T>| {
                        x.clone().rows()
                            .zip(y.clone().rows())
                            .map(|(x_test, y_test)| {
                                let x_modified = x_test.prepend_one().to_ket();
                                let value = dot(&w, &x_modified).to_scalar() - *y_test.get_v(index);
                                if cost_function == CostFunction::Abs { 
                                    T::abs(value)
                                } else {
                                    T::powi(value, 2) 
                                }
                            })
                            .sum()
                    }) as Box<dyn Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> T+ Send + Sync>
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use super::{ LinearRegression, Tensor, CostFunction, GradientDescent};

    fn generate_x(count: usize, x_min: f32, x_max: f32) -> Vec<Vec<f32>>{
        let mut rng = rand::thread_rng();
        (0..count)
            .map(|_| vec![
                rng.gen_range(x_min..x_max),
                rng.gen_range(x_min..x_max)
            ])
            .collect()
    }

    fn calc_y(x: &Vec<Vec<f32>>) -> Vec<Vec<f32>>{
        let mut rng = rand::thread_rng();
        x.iter()
            .map(|x_vec| vec![
                2.5 * x_vec[0] + 3.5* x_vec[1]  + 1.7 + rng.gen_range(-1.0..1.0),
                0.8 * x_vec[0] - 1.2* x_vec[1]  + 0.5 + rng.gen_range(-1.0..1.0),
               -1.1 * x_vec[0] - 0.5* x_vec[1]  - 2.2 + rng.gen_range(-1.0..1.0),
            ])
            .collect()
    }

    fn create_train_test(train_size: usize, test_size: usize) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>, Tensor<f32>) {
        let x_train = generate_x(train_size, 0.0, 10.0);
        let y_train = calc_y(&x_train);
        let x_test = generate_x(test_size, 0.0, 10.0);
        let y_test = calc_y(&x_test);

        let x_train = Tensor::matrix(x_train);
        let y_train = Tensor::matrix(y_train);
        let x_test = Tensor::matrix(x_test);
        let y_test = Tensor::matrix(y_test);
        (x_train, y_train, x_test, y_test)
    }

            
    #[test]
    fn new() {
        let train_size: usize = 10;
        let test_size: usize = 5;
        let (x_train, y_train, x_test, y_test) = create_train_test(train_size, test_size);

        assert_eq!(x_train.shape, vec![train_size, 2]);
        assert_eq!(y_train.shape, vec![train_size, 3]);
        assert_eq!(x_test.shape, vec![test_size, 2]);
        assert_eq!(y_test.shape, vec![test_size, 3]);

        let mut model = LinearRegression {
            cost_function: CostFunction::LeastSquares,
            optimizator: GradientDescent {
                step_count: 1000,
                step_size: 3.0,
                ..Default::default()
            },
            ..Default::default()
        };
        model.fit(x_train, y_train);
        let y_predict = model.predict(x_test);

        y_predict.rows()
            .zip(y_test.rows())
            .for_each(|(predict, test)| {
                assert!(predict.is_near(&test, 2.0), "{:?} vs {:?}", predict.data, test.data);
        });
    }
}
