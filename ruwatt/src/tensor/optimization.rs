use num::Float;
use super::Tensor;
use super::gradient;
use std::fmt::Debug;


pub fn gradient_descent<T>(
    f: &dyn Fn(&Tensor<T>) -> T, 
    w0: Tensor<T>, step_count: i8, step_size: f32, decrement: bool) -> T where T: Float + Debug{
    let mut w = w0;
    for step in 0..step_count {
        let grad = gradient(f, &w);
        w = &w - &grad;
        println!("w: {:?}", w);
    }
    f(&w)
  }
