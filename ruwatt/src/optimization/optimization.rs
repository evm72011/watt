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

impl<T> OptimizationResult<T> where T: Float + Debug {
    fn new(value: T, arg: Tensor<T>) -> Self {
        Self { 
            value, 
            arg 
        }
    }
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

    fn get_optimal(&mut self) -> Option<OptimizationResult<T>> {
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
    pub start_point: Tensor<T>, 
    pub step_count: i16,
    pub betta: T, 
    pub step_size: T, 
    pub decrement_step: bool,
    pub analyze_history: bool,
    pub small_gradient_value: T,
    pub small_gradient_behaviour: SmallGradientBehaviour,
    pub derivative_delta: T,
    pub progress: OptimizationProgress<T>,
    pub result: Option<OptimizationResult<T>>,
    pub logs: Vec<String>
}

impl<'a, T> Default for GradientDescent<'a, T> where T: Float + Debug {
    fn default() -> Self {
        Self {
            func: &|_| T::zero(),
            start_point: Tensor::<T>::vector(&[T::zero()]),
            step_count: 1000,
            betta: T::from(0.7).unwrap(), 
            step_size: T::one(), 
            decrement_step: true,
            analyze_history: true,
            small_gradient_value: T::from(0.0001).unwrap(),
            small_gradient_behaviour: SmallGradientBehaviour::Interrupt,
            derivative_delta: T::from(0.0001).unwrap(),
            progress: OptimizationProgress::new(),
            result: None,
            logs: vec![]
        }
    }
}

impl<'a, T> GradientDescent<'a, T> where T: Float + Debug {
    pub fn run(&mut self) {
        let now = Instant::now();
        let mut arg = self.start_point.clone();
        let mut grad_prev = Tensor::<T>::zeros(arg.shape.to_vec());
        let result = OptimizationResult::new((self.func)(&arg), arg.clone());
        self.progress.init(result);
        for step in 0..self.step_count {
            let step_f = T::from(step + 1).unwrap();
            let size = if self.decrement_step { self.step_size / step_f} else { self.step_size };
    
            let mut grad = gradient(self.func, &arg, self.derivative_delta);
            if  grad.is_small(self.small_gradient_value) {
                match &self.small_gradient_behaviour {
                    SmallGradientBehaviour::Displace => {
                        arg = Tensor::random(arg.shape.to_vec());
                        self.logs.push(format!("Step {}: Small gradient. Change point.", step));
                        continue;
                    }
                    SmallGradientBehaviour::Interrupt => {
                        self.logs.push(format!("Step {}: Small gradient. Interrupted.", step));
                        break;
                    }
                    _other => {}
                }
            }
            grad.set_length(size);
    
            grad = if step == 0 { 
                grad 
            } else { 
                &grad_prev * &self.betta + &grad * &(T::one() - self.betta)
            };
            grad_prev = grad.clone();
            arg = arg - grad;
    
            let result = OptimizationResult::new((self.func)(&arg), arg.clone());
            if self.analyze_history {
                self.progress.add(result);
            } else {
                self.progress.init(result);
            }
        }
        self.result = self.progress.get_optimal();
        self.logs.push(format!("Elapsed: {:.2?}.", now.elapsed()));
    }
}
