use num::Float;
use crate::{assert_matrix, optimization::GradientDescent};
use crate::tensor::{Tensor, dot::dot};

pub struct LinearRegression<T=f32> where T: Float {
    pub coef: Tensor<T>
}

impl<T> Default for LinearRegression<T> where T: Float  {
    fn default() -> Self {
        Self {
            coef: Tensor::zeros(vec![1])
        }
    }
}

fn least_squares<T>(_x: Tensor<T>, _y: Tensor<T>) where T: Float {

}

impl<T> LinearRegression<T> where T: Float {
    pub fn fit(&mut self, x: Tensor<T>, y: Tensor<T>) {
        assert_matrix!(x);
        assert_matrix!(y);
        assert_eq!(x.row_count(), y.row_count(), "Count of x train not correspond to y");

        let closures: Vec<Box<dyn Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> T >> = 
            (0..(y.col_count()))
                .map(|index| {
                    Box::new(move |w: &Tensor<T>, x: &Tensor<T>, y: &Tensor<T>| {
                        let mut result = T::zero();
                        x.clone().rows()
                            .zip(y.clone().rows())
                            .map(|(x_test, y_test)| {
                                let x_modified = x_test.add_one().to_ket();
                                T::abs(dot(&w, &x_modified).to_scalar() - *y_test.get_v(index))
                            })
                            .for_each(|val| {
                                result = result + val;
                            });
                        result
                    }) as Box<dyn Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> T>
            })
            .collect();
        
        let mut step = 0;
        for closure in closures {
            let x_clone = x.clone();
            let y_clone = y.clone();
            let f = move |w: &Tensor<T>| {
                closure(w, &x_clone, &y_clone)
            };
            let mut optimizator = GradientDescent {
                func: &f,
                start_point: Tensor::bra(vec![T::one(); x.shape[1] + 1]),
                derivative_delta: T::from(0.000001).unwrap(),
                step_size: T::from(3.0).unwrap(),
                step_count: 1_000,
                ..Default::default()
            };
            optimizator.run();
            println!("step: {step}");
            let result = optimizator.result.unwrap();
            let arg: Vec<f32> = result.arg.data.iter().map(|v| v.to_f32().unwrap()).collect();
            println!("result.arg: {:?}", arg);
            println!("result.value: {:?}", result.value.to_f32());
            println!("logs: {:?}", optimizator.logs);
            if step == 0 {
                self.coef = result.arg.to_ket();
            } else {
                self.coef.append_col(result.arg.to_ket())
            }
            step += 1;
            println!("----------------------------------");
        }
    }

    pub fn predict(&mut self, x: Tensor<T>) -> Tensor<T> {
        let x_modified = x.add_one().to_ket();
        let w = self.coef.tr();
        dot(&w, &x_modified)
    }
}
