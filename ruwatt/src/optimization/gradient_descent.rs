use num::Float;
use std::{fmt::Debug, iter::Sum};
use crate::tensor::{ dot, Tensor, Vector };
use super::{gradient, hessian, ResultEntry, ResultLogs};

#[derive(Clone)]
pub struct GradientDescent<'a, T> where T: Float + Debug {
    pub func: &'a dyn Fn(&Tensor<T>) -> T,
    pub grad_func: Option<&'a dyn Fn(&Tensor<T>) -> Tensor<T>>,
    pub hessian: Option<&'a dyn Fn(&Tensor<T>) -> Tensor<T>>,
    pub start_point: Tensor<T>, 
    pub step_count: i16,
    pub betta: T, 
    pub step_size: StepSize<T>,
    pub analyze_progress: bool,
    pub derivative_delta: T,
    pub results: ResultLogs<T>,
    pub result: Option<ResultEntry<T>>,
    pub grad_prev: Tensor<T>
}

#[derive(Clone)]
pub enum StepSize<T> where T: Float {
    OriginGrad,
    Fixed(T),
    Decrement(T),
    Newton
}

impl<'a, T> Default for GradientDescent<'a, T> where T: Float + Debug {
    fn default() -> Self {
        Self {
            func: &|_| T::zero(),
            grad_func: None,
            hessian: None,
            start_point: Vector::ket(vec![T::zero()]),
            step_count: 1000,
            betta: T::from(0.7).unwrap(), 
            step_size: StepSize::Decrement(T::one()), 
            analyze_progress: true,
            derivative_delta: T::from(0.0001).unwrap(),
            results: ResultLogs::new(),
            result: None,
            grad_prev: Vector::ket(vec![T::zero()])
        }
    }
}

impl<'a, T> GradientDescent<'a, T> where T: Float + Sum + Debug {
    pub fn run(&mut self) {
        let mut arg = self.start_point.clone();
        self.save_result((self.func)(&arg), arg.clone());
        for step in 0..self.step_count {
            let grad = match self.grad_func {
                Some(grad_func) => grad_func(&arg),
                None => gradient(self.func, &arg, self.derivative_delta)
            };
            let grad = self.set_grad_length(grad, step, &arg);
            arg = arg - grad;
            self.save_result((self.func)(&arg), arg.clone());
        }
        self.result = self.results.get_optimal_result();
    }

    fn set_grad_length(&mut self, grad: Tensor<T>, step: i16, arg: &Tensor<T>) -> Tensor<T> {
        let gradient = match self.step_size {
            StepSize::OriginGrad => grad,
            StepSize::Fixed(size) => grad.set_length(size),
            StepSize::Decrement(size) => {
                let size = size / T::from(step + 1).unwrap();
                grad.set_length(size)
            },
            StepSize::Newton => {
                let hessian = match self.hessian {
                    Some(hessian) => hessian(arg),
                    None => hessian(self.func, arg, self.derivative_delta)
                };
                println!("arg: {:?}", arg.data);
                println!("grad: {:?}", grad.data);
                println!("hess: {:?}", hessian.data);
                println!("hess.inverse: {:?}", hessian.inverse().unwrap().data);
                println!("dot: {:?}", dot(&hessian.inverse().unwrap(), &grad).data);
                println!("-------------------------------------------");
                dot(&hessian.inverse().unwrap(), &grad)
            }
        };
        self.apply_momentum_acceleration(gradient, step)
    }

    fn save_result(&mut self, value: T, arg: Tensor<T>) {
        let result = ResultEntry { 
            value, 
            arg 
        };
        if self.analyze_progress {
            self.results.add(result);
        } else {
            self.results.init(result);
        }
    }

    fn apply_momentum_acceleration(&mut self, grad: Tensor<T>, step: i16) -> Tensor<T>{
        let result = if step == 0 { 
            grad
        } else { 
            &self.grad_prev * &self.betta + &grad * &(T::one() - self.betta)
        };
        self.grad_prev = result.clone();
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_near;
    use crate::tensor::Vector;
    use super::{Tensor, GradientDescent, StepSize};

    fn f(x: &Tensor) -> f64 {
        2.0 + x.get_v(0).powi(2) + x.get_v(1).powi(2)
    }

    fn grad_f(vector: &Tensor) -> Tensor {
        let w0 = vector.get_v(0);
        let w1 = vector.get_v(1);
        Vector::ket(vec![2.0*w0, 2.0*w1])
    }

    #[test]
    fn gradient_descent_analytic_grad() {
        let mut optimizator = GradientDescent {
            func: &f,
            grad_func: Some(&grad_f),
            start_point: Vector::ket(vec![3.0, 3.0]),
            ..Default::default()
        };
        optimizator.run();
        let result = optimizator.result.unwrap();
        let arg_expected = Vector::ket(vec![0.0, 0.0]);
        assert!(f64::abs(result.value - 2.0) < 0.001);
        assert_near!(result.arg, arg_expected, 0.001)
    }

    #[test]
    fn gradient_descent_numeric_grad() {
        let mut optimizator = GradientDescent {
            func: &f,
            start_point: Vector::ket(vec![3.0, 3.0]),
            ..Default::default()
        };
        optimizator.run();
        let result = optimizator.result.unwrap();
        let arg_expected = Vector::ket(vec![0.0, 0.0]);
        assert!(f64::abs(result.value - 2.0) < 0.001);
        assert_near!(result.arg, arg_expected, 0.001)
    }

    #[ignore]
    #[test]
    fn gradient_descent_newton() {
        let mut optimizator = GradientDescent {
            func: &f,
            start_point: Vector::ket(vec![3.0, 3.0]),
            step_size: StepSize::Newton,
            step_count: 1,
            ..Default::default()
        };
        optimizator.run();
        let result = optimizator.result.unwrap();
        let arg_expected = Vector::ket(vec![0.0, 0.0]);
        println!("{}", result.value);
        assert!(f64::abs(result.value - 2.0) < 0.01);
        assert_near!(result.arg, arg_expected, 0.01)
    }
}
