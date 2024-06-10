use num::Float;
use super::gradient;
use super::super::Tensor;
use std::cmp::Ordering;
use std::time::Instant;

#[derive(Clone)]
pub struct OptimizationResult<T> where T: Float {
    pub value: T,
    pub arg: Tensor<T>
}

#[derive(Clone)]
pub struct OptimizationProgress<T> where T: Float {
    pub data: Vec<OptimizationResult<T>>
}

impl<T> OptimizationProgress<T> where T: Float {
    fn new() -> Self {
        Self { 
            data: vec![]
        }
    }

    fn add(&mut self, result: OptimizationResult<T>) {
        self.data.push(result);
    }

    fn init(&mut self, result: OptimizationResult<T>) {
        self.data = vec![result];
    }

    fn get_optimal_result(&mut self) -> Option<OptimizationResult<T>> {
        self.data.iter().min_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal)).cloned()
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub enum SmallGradientBehaviour {
    Ignore,
    Interrupt,
    Displace
}

#[derive(Clone)]
pub struct GradientDescent<'a, T> where T: Float {
    pub func: &'a dyn Fn(&Tensor<T>) -> T,
    pub grad_func: Option<&'a dyn Fn(&Tensor<T>) -> Tensor<T>>,
    pub start_point: Tensor<T>, 
    pub step_count: i16,
    pub betta: T, 
    pub step_size: T, 
    pub decrement_step: bool,
    pub analyze_progress: bool,
    pub small_gradient_behaviour: SmallGradientBehaviour,
    pub derivative_delta: T,
    pub progress: OptimizationProgress<T>,
    pub result: Option<OptimizationResult<T>>,
    pub logs: Vec<String>,
    pub grad_prev: Tensor<T>, 
}

impl<'a, T> Default for GradientDescent<'a, T> where T: Float {
    fn default() -> Self {
        Self {
            func: &|_| T::zero(),
            grad_func: None,
            start_point: Tensor::<T>::ket(vec![T::zero()]),
            step_count: 1000,
            betta: T::from(0.7).unwrap(), 
            step_size: T::one(), 
            decrement_step: true,
            analyze_progress: true,
            small_gradient_behaviour: SmallGradientBehaviour::Interrupt,
            derivative_delta: T::from(0.0001).unwrap(),
            progress: OptimizationProgress::new(),
            result: None,
            logs: vec![],
            grad_prev: Tensor::<T>::ket(vec![T::zero()])
        }
    }
}

impl<'a, T> GradientDescent<'a, T> where T: Float {
    pub fn run(&mut self) {
        let now = Instant::now();
        let mut arg = self.start_point.clone();
        self.save_result((self.func)(&arg), arg.clone());
        let mut step_counter = 0;
        for step in 0..self.step_count {
            step_counter += 1;
            let size = self.calc_step_size(step);
            let mut grad = match self.grad_func {
                Some(grad_func) => grad_func(&arg),
                None => gradient(self.func, &arg, self.derivative_delta)
            };
            let delta = T::from(0.0001).unwrap();
            match &self.small_gradient_behaviour {
                SmallGradientBehaviour::Displace => {
                    if grad.is_small(delta) {
                        arg = Tensor::random(arg.shape.to_vec());
                        self.logs.push(format!("Step {}: Small gradient. Change point.", step));
                        continue;                        
                    }
                }
                SmallGradientBehaviour::Interrupt => {
                    if grad.is_small(delta) {
                        self.logs.push(format!("Step {}: Small gradient. Interrupted.", step));
                        break;
                    }
                }
                _other => {}
            }

            grad.set_length(size);
            grad = self.apply_momentum_acceleration(grad, step); 
            arg = arg - grad;
    
            self.save_result((self.func)(&arg), arg.clone());
        }
        self.result = self.progress.get_optimal_result();
        self.logs.push(format!("Elapsed: {:.2?} and {} steps.", now.elapsed(), step_counter));
    }

    fn save_result(&mut self, value: T, arg: Tensor<T>) {
        let result = OptimizationResult { 
            value, 
            arg 
        };
        if self.analyze_progress {
            self.progress.add(result);
        } else {
            self.progress.init(result);
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

    fn calc_step_size(&self, step: i16) -> T {
        let step_f = T::from(step + 1).unwrap();
        if self.decrement_step { 
            self.step_size / step_f
        } else { 
            self.step_size 
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, GradientDescent};

    fn f(x: &Tensor) -> f32 {
        2.0 + x.get_v(0).powi(2) + x.get_v(1).powi(2)
    }

    fn grad_f(vector: &Tensor) -> Tensor {
        let w0 = vector.get_v(0);
        let w1 = vector.get_v(1);
        Tensor::ket(vec![2.0*w0, 2.0*w1])
    }

    #[test]
    fn gradient_descent_analytic_grad() {
        let mut optimizator = GradientDescent {
            func: &f,
            grad_func: Some(&grad_f),
            start_point: Tensor::ket(vec![3.0, 3.0]),
            ..Default::default()
        };
        optimizator.run();
        let result = optimizator.result.unwrap();
        let arg_expected = Tensor::ket(vec![0.0, 0.0]);
        assert!(f32::abs(result.value - 2.0) < 0.001);
        assert!(result.arg.is_ket());
        assert!(result.arg.is_near(&arg_expected, 0.001))
    }

    #[test]
    fn gradient_descent_numeric_grad() {
        let mut optimizator = GradientDescent {
            func: &f,
            start_point: Tensor::ket(vec![3.0, 3.0]),
            ..Default::default()
        };
        optimizator.run();
        let result = optimizator.result.unwrap();
        let arg_expected = Tensor::ket(vec![0.0, 0.0]);
        assert!(f32::abs(result.value - 2.0) < 0.001);
        assert!(result.arg.is_ket());
        assert!(result.arg.is_near(&arg_expected, 0.001))
    }
}