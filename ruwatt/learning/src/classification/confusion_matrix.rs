use std::collections::{BTreeSet, HashMap};
use std::fmt::{Formatter, Result, Display};
use num::Float;
use tensor::{assert_shape, Matrix, Tensor};

pub struct ConfusionMatrix {
    data: Vec<Vec<i32>>,
    values: Vec<i32>
}

fn unique_values(vec1: &Vec<i32>, vec2: &Vec<i32>) -> Vec<i32> {
    let values = [vec1.clone(), vec2.clone()].concat();
    let set: BTreeSet<_> = values.iter().cloned().collect(); 
    set.into_iter().collect()
}

fn to_vec_i32<T>(tensor: &Tensor<T>) -> Vec<i32> where T: Float {
    tensor.data.iter().map(|&x| x.to_i32().unwrap()).collect()
}

impl ConfusionMatrix {
    pub fn new<T>(actual: &Tensor<T>, predicted: &Tensor<T>) -> Self where T: Float {
        assert_shape!(actual, predicted);
        let a_data = to_vec_i32(actual);
        let p_data = to_vec_i32(predicted);
        let values = unique_values(&a_data, &p_data);
        let mut data = Tensor::<T>::zeros(vec![values.len(), values.len()]);
        let value_to_index: HashMap<_, _> = values.iter().enumerate().map(|(index, value)| (value, index)).collect();

        a_data.iter()
            .zip(p_data.iter())
            .for_each(|(a_value, p_value)| {
                let a_index = value_to_index.get(a_value).unwrap();
                let p_index = value_to_index.get(p_value).unwrap();
                let indices = vec![*a_index, *p_index];
                data.set(indices.clone(), data.get(indices) + T::one())
            });
        let data: Vec<_> = data.rows()
            .map(|row| row.data.iter().map(|val| val.to_i32().unwrap()).collect::<Vec<_>>())
            .collect();

        ConfusionMatrix { values, data }
    }

    pub fn to_tensor<T>(&self) -> Tensor<T> where T: Float {
        let data: Vec<_> = self.data.iter()
            .map(|row| row.iter().map(|value| T::from(*value).unwrap()).collect::<Vec<_>>())
            .collect();
        Matrix::new(data)
    }
}

fn pad_string(input: String, length: usize) -> String {
    if input.len() >= length {
        input.to_string()
    } else {
        let padding = " ".repeat(length - input.len());
        format!("{}{}", input, padding)
    }
} 

fn print_vec<T>(vector: Vec<T>, f: &mut Formatter) -> Result where T: Display  {
    let vector: Vec<_> = vector.iter()
        .map(|value| pad_string(value.to_string(), 10))
        .collect();
    writeln!(f, "{}", vector.join(" "))?;
    Ok(())   
}

fn print_header(values: &Vec<i32>, f: &mut Formatter) -> Result {
    let mut vector: Vec<_> = values.iter().map(|v| v.to_string()).collect();
    vector.insert(0, pad_string(String::from("act\\pred"), 10));
    print_vec(vector, f)
}

fn print_row(value: i32, values: &Vec<i32>, f: &mut Formatter)-> Result {
    let mut vector = values.clone();
    vector.insert(0, value);
    print_vec(vector, f)
}

impl Display for ConfusionMatrix {
    fn fmt(&self, f: &mut Formatter) -> Result {
        print_header(&self.values, f)?;
        self.values.iter()
            .zip(self.data.iter())
            .for_each(|(&value, values)| print_row(value, values, f).unwrap());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tensor::{Matrix, Vector};
    use super::ConfusionMatrix;

    #[test]
    fn confusion_matrix_test() {
        let a = Vector::ket(vec![0.0, 1.0, 0.0, 1.0, 0.0 ]);
        let p = Vector::ket(vec![0.0, 1.0, 0.0, 1.0, 1.0 ]);
        let expected = Matrix::new(vec![
            vec![2.0, 1.0],
            vec![0.0, 2.0]
        ]);
        let recieved = ConfusionMatrix::new(&a, &p).to_tensor();
        assert_eq!(recieved, expected)
    }
}