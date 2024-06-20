use num::Float;
use std::iter::Sum;

use crate::{assert_matrix, optimization::GradientDescent};
use crate::tensor::{ dot, Tensor, Vector };

#[derive(PartialEq, Clone)]
pub enum CostFunction {
    LeastSquares,
    Abs,
}

pub struct LinearRegression<'a, T=f32> where T: Float + Sum {
    pub trained: bool,
    pub feature_count: usize,
    pub coef: Tensor<T>,
    pub cost_function: CostFunction,
    pub optimizator: GradientDescent<'a, T>
}

impl<'a, T> Default for LinearRegression<'a, T> where T: Float + Sum {
    fn default() -> Self {
        Self {
            trained: false,
            feature_count: 0,
            coef: Tensor::empty(),
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
        self.feature_count = x.col_count();

        let closures = self.create_closures(y.col_count());
        for closure in closures.iter() {
            let f = |w: &Tensor<T>| closure(w, &x, &y);
            let mut optimizator = GradientDescent {
                func: &f,
                start_point: Vector::bra(vec![T::one(); x.col_count() + 1]),
                ..self.optimizator.clone()
            };
            optimizator.run();
            let result = optimizator.result.unwrap();
            self.coef.append_row(result.arg);
        }
        self.trained = true;
    }

    pub fn predict(&mut self, x: Tensor<T>) -> Tensor<T> {
        assert!(self.trained, "Model is not trained");
        assert_eq!(self.feature_count, x.col_count(), "Feature count must be {}", self.feature_count);
        let mut result = Tensor::empty();
        x.rows().for_each(|item| {
            let x_modified = item.prepend_one().to_ket();
            let row = dot(&self.coef, &x_modified).to_bra();
            result.append_row(row)
        });
        result
    }

    fn create_closures(&self, count: usize) -> Vec<Box<dyn Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> T + Send + Sync>>{
        (0..count)
                .map(|index| {
                    let cost_function = self.cost_function.clone();
                    Box::new(move |w: &Tensor<T>, x: &Tensor<T>, y: &Tensor<T>| {
                        x.clone().rows()
                            .zip(y.clone().rows())
                            .map(|(x_test, y_test)| {
                                let x_modified = x_test.prepend_one().to_ket();
                                let value = dot(&w, &x_modified).to_scalar() - y_test.get_v(index);
                                if cost_function == CostFunction::Abs { 
                                    T::abs(value)
                                } else {
                                    T::powi(value, 2) 
                                }
                            })
                            .sum()
                    }) as Box<dyn Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> T + Send + Sync>
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use crate::tensor::{ Matrix, Tensor };
    use super::{ LinearRegression, CostFunction, GradientDescent};

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

        let x_train = Matrix::new(x_train);
        let y_train = Matrix::new(y_train);
        let x_test = Matrix::new(x_test);
        let y_test = Matrix::new(y_test);
        (x_train, y_train, x_test, y_test)
    }

            
    #[test]
    fn linear_regression() {
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
