use num::Float;
use super::gradient;
use super::super::Tensor;
use std::fmt::Debug;
use std::cmp::Ordering;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct OptimizationResult<T> where T: Float + Debug {
    pub value: T,
    pub arg: Tensor<T>
}

pub struct OptimizationProgress<T> where T: Float + Debug {
    pub data: Vec<OptimizationResult<T>>
}

impl<T> OptimizationProgress<T> where T: Float + Debug {
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
pub enum SmallGradientBehaviour {
    Ignore,
    Interrupt,
    Displace
}

pub struct GradientDescent<'a, T> where T: Float + Debug {
    pub func: &'a dyn Fn(&Tensor<T>) -> T,
    pub grad_func: Option<&'a dyn Fn(&Tensor<T>) -> Tensor<T>>,
    pub start_point: Tensor<T>, 
    pub step_count: i16,
    pub betta: T, 
    pub step_size: T, 
    pub decrement_step: bool,
    pub analyze_progress: bool,
    pub small_gradient_value: T,
    pub small_gradient_behaviour: SmallGradientBehaviour,
    pub derivative_delta: T,
    pub progress: OptimizationProgress<T>,
    pub result: Option<OptimizationResult<T>>,
    pub logs: Vec<String>,
    pub grad_prev: Tensor<T>, 
}

impl<'a, T> Default for GradientDescent<'a, T> where T: Float + Debug {
    fn default() -> Self {
        Self {
            func: &|_| T::zero(),
            grad_func: None,
            start_point: Tensor::<T>::vector(vec![T::zero()]),
            step_count: 1000,
            betta: T::from(0.7).unwrap(), 
            step_size: T::one(), 
            decrement_step: true,
            analyze_progress: true,
            small_gradient_value: T::from(0.0001).unwrap(),
            small_gradient_behaviour: SmallGradientBehaviour::Interrupt,
            derivative_delta: T::from(0.0001).unwrap(),
            progress: OptimizationProgress::new(),
            result: None,
            logs: vec![],
            grad_prev: Tensor::<T>::vector(vec![T::zero()])
        }
    }
}

impl<'a, T> GradientDescent<'a, T> where T: Float + Debug {
    pub fn run(&mut self) {
        let now = Instant::now();
        let mut arg = self.start_point.clone();
        self.save_result((self.func)(&arg), arg.clone());
        for step in 0..self.step_count {
            let size = self.calc_step_size(step);
            let mut grad = match self.grad_func {
                Some(grad_func) => grad_func(&arg),
                None => gradient(self.func, &arg, self.derivative_delta)
            };
            match &self.small_gradient_behaviour {
                SmallGradientBehaviour::Displace => {
                    if grad.is_small(self.small_gradient_value) {
                        arg = Tensor::random(arg.shape.to_vec());
                        self.logs.push(format!("Step {}: Small gradient. Change point.", step));
                        continue;                        
                    }
                }
                SmallGradientBehaviour::Interrupt => {
                    if grad.is_small(self.small_gradient_value) {
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
        self.logs.push(format!("Elapsed: {:.2?}.", now.elapsed()));
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
