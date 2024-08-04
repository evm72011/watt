use num::Float;
use tensor::Tensor;
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

    pub fn add_if_optimal(&mut self, result: ResultEntry<T>) {
        if self.is_empty() || self.data[0].value > result.value {
            self.add(result);
        }
    }

    pub fn get_optimal_result(&mut self) -> Option<ResultEntry<T>> {
        self.data.iter()
            .min_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal))
            .cloned()
    }

    fn is_empty(&self) -> bool {
        self.data.len() == 0
    }
}
