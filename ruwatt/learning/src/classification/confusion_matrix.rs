use std::{collections::{BTreeSet, HashMap}, fmt::Debug};

use data_frame::DataFrame;
use num::Float;
use tensor::{assert_shape, Tensor};

fn unique_values<T: Clone + Ord>(vec: Vec<T>) -> Vec<T> {
    let set: BTreeSet<_> = vec.iter().cloned().collect(); 
    set.into_iter().collect()
}

pub fn confusion_matrix<T>(a: &Tensor<T>, b: &Tensor<T>) -> DataFrame<T> where T: Float + Debug + Default {
    assert_shape!(a, b);
    
    let a_data: Vec<i32> = a.data.iter().map(|&x| x.to_i32().unwrap()).collect();
    let b_data: Vec<i32> = b.data.iter().map(|&x| x.to_i32().unwrap()).collect();

    let values = unique_values([a_data.clone(), b_data.clone()].concat());
    let mut data = Tensor::zeros(vec![values.len(), values.len()]);
    let value_to_index: HashMap<_, _> = values.iter().enumerate().map(|(i, v)| (v, i)).collect();
    
    a_data.iter()
        .zip(b_data.iter())
        .for_each(|(a_value, b_value)| {
            let a_index = value_to_index.get(a_value).unwrap();
            let b_index = value_to_index.get(b_value).unwrap();
            let indices = vec![*b_index, *a_index];
            data.set(indices.clone(), data.get(indices) + T::one())
        });
    DataFrame::from_tensor(&data)
}

#[cfg(test)]
mod tests {
    use tensor::{Matrix, Vector};

    use super::confusion_matrix;

    #[test]
    fn confusion_matrix_test() {
        let a = Vector::ket(vec![0.0, 1.0, 0.0, 1.0, 0.0 ]);
        let b = Vector::ket(vec![0.0, 1.0, 0.0, 1.0, 1.0 ]);
        let expected = Matrix::new(vec![
            vec![2.0, 0.0],
            vec![1.0, 2.0]
        ]);
        let recieved = confusion_matrix(&a, &b).to_tensor(None);
        assert_eq!(recieved, expected)
    }
}