use std::collections::{BTreeSet, HashMap};

use num::Float;
use tensor::{assert_shape, Tensor};

fn unique_values<T: Clone + Ord>(vec: Vec<T>) -> Vec<T> {
    let set: BTreeSet<_> = vec.iter().cloned().collect(); 
    set.into_iter().collect()
}

pub fn confusion_matrix<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<f32> where T: Float {
    assert_shape!(a, b);
    
    let a_data: Vec<i32> = a.data.iter().map(|&x| x.to_i32().unwrap()).collect();
    let b_data: Vec<i32> = b.data.iter().map(|&x| x.to_i32().unwrap()).collect();

    let values = unique_values([a_data.clone(), b_data.clone()].concat());
    let mut result = Tensor::zeros(vec![values.len(), values.len()]);
    let value_to_index: HashMap<_, _> = values.iter().enumerate().map(|(i, v)| (v, i)).collect();
    
    a_data.iter()
        .zip(b_data.iter())
        .for_each(|(a_value, b_value)| {
            let a_index = value_to_index.get(a_value).unwrap();
            let b_index = value_to_index.get(b_value).unwrap();
            let indices = vec![*a_index, *b_index];
            result.set(indices.clone(), result.get(indices) + 1.0)
        });
    result
}
