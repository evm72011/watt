use num::Float;
use crate::tensor::Tensor;
use std::cmp::Ordering;

#[derive(Clone)]
pub struct ResultEntry<T> where T: Float {
    pub value: T,
    pub arg: Tensor<T>
}

#[derive(Clone)]
pub struct ResultLogs<T> where T: Float {
    pub data: Vec<ResultEntry<T>>
}

impl<T> ResultLogs<T> where T: Float {
    pub fn new() -> Self {
        Self { 
            data: vec![]
        }
    }

    pub fn add(&mut self, result: ResultEntry<T>) {
        self.data.push(result);
    }

    pub fn init(&mut self, result: ResultEntry<T>) {
        self.data = vec![result];
    }

    pub fn get_optimal_result(&mut self) -> Option<ResultEntry<T>> {
        self.data.iter().min_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal)).cloned()
    }
}
