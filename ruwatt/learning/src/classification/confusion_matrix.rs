use std::{collections::{BTreeSet, HashMap}, fmt::Debug};

use data_frame::{DataFrame, FrameDataCell, FrameHeader};
use num::Float;
use tensor::{assert_shape, Matrix, Tensor};

fn unique_values<T: Clone + Ord>(vec: Vec<T>) -> Vec<T> {
    let set: BTreeSet<_> = vec.iter().cloned().collect(); 
    set.into_iter().collect()
}

fn titel_column<T>(values: &Vec<i32>) -> DataFrame<T> where T: Float + Debug + Default {
    
    let data: Vec<_> = values.iter()
        .map(|val| vec![T::from(*val).unwrap()])
        .collect();
    let matrix = Matrix::new(data);
    let mut result = DataFrame::from_tensor(&matrix);
    result.headers[0].name = String::from("a\\p");
    result
}

fn headers<T>(values: &Vec<i32>) -> Vec<FrameHeader<T>> where T: Float + Default {
    let mut result =  values.iter()
            .map(|val| FrameHeader {
                name: val.to_string(),
                data_type: FrameDataCell::<T>::Number(Default::default())
            })
            .collect::<Vec<_>>();
    let first_header = FrameHeader {
        name: String::from("a\\p"),
        data_type: FrameDataCell::<T>::Number(Default::default())
    };
    result.insert(0, first_header);
    result
}

pub fn confusion_matrix<T>(actual: &Tensor<T>, predicted: &Tensor<T>) -> DataFrame<T> where T: Float + Debug + Default {
    assert_shape!(actual, predicted);
    
    let a_data: Vec<i32> = actual.data.iter().map(|&x| x.to_i32().unwrap()).collect();
    let p_data: Vec<i32> = predicted.data.iter().map(|&x| x.to_i32().unwrap()).collect();

    let values = unique_values([a_data.clone(), p_data.clone()].concat());
    let mut data = Tensor::zeros(vec![values.len(), values.len()]);
    let value_to_index: HashMap<_, _> = values.iter().enumerate().map(|(index, value)| (value, index)).collect();
    
    a_data.iter()
        .zip(p_data.iter())
        .for_each(|(a_value, p_value)| {
            let a_index = value_to_index.get(a_value).unwrap();
            let p_index = value_to_index.get(p_value).unwrap();
            let indices = vec![*a_index, *p_index];
            data.set(indices.clone(), data.get(indices) + T::one())
        });
    let mut result = titel_column(&values);
    result.append_cols(&DataFrame::from_tensor(&data));
    result.headers = headers(&values);
    result
}

#[cfg(test)]
mod tests {
    use tensor::{Matrix, Vector};

    use super::confusion_matrix;

    #[test]
    fn confusion_matrix_test() {
        let a = Vector::ket(vec![0.0, 1.0, 0.0, 1.0, 0.0 ]);
        let p = Vector::ket(vec![0.0, 1.0, 0.0, 1.0, 1.0 ]);
        let expected = Matrix::new(vec![
            vec![0.0,   2.0, 1.0],
            vec![1.0,   0.0, 2.0]
        ]);
        let recieved = confusion_matrix(&a, &p).to_tensor(None);
        assert_eq!(recieved, expected)
    }
}