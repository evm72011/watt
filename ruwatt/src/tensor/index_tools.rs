use num::Float;
use std::marker::PhantomData;

pub struct IndexTools<T> where T: Float {
    _marker: PhantomData<T>
}

impl<T> IndexTools<T> where T: Float {
    fn check_range(indices: &Vec<usize>, shape: &Vec<usize>) {
        assert!(
            shape.len() == indices.len() && 
            shape.iter().zip(indices.iter()).any(|(a, b)| a > b),
            "Index out of range: shape = {:?}, indices = {:?}",
            shape,
            indices
        );
    }
    
    fn calc_index(indices: Vec<usize>, shape: &Vec<usize>) -> usize {
        Self::check_range(&indices, shape );
        let mut index = 0;
        let mut stride = 1;
        for (i, &dim) in shape.iter().rev().enumerate() {
            index += indices[shape.len() - 1 - i] * stride;
            stride *= dim;
        }
        index
    }
  
    pub fn get_item(indices: Vec<usize>, shape: &Vec<usize>, data: &Vec<T>) -> T {
        let index = Self::calc_index(indices, shape );
        data[index]
    }
  
    pub fn set_item(indices: Vec<usize>, value: T, shape: &Vec<usize>, data: &mut Vec<T>) {
        let index = Self::calc_index(indices, shape );
        data[index] = value;
    }

    pub fn get_row_count(shape: &Vec<usize>) -> usize {
        shape[0]
    }

    pub fn get_col_count(shape: &Vec<usize>) -> usize {
        shape[1]
    }

    pub fn get_row(index: usize, shape: &Vec<usize>, data: &Vec<T>) -> Vec<T> {
        let row_count = Self::get_row_count(shape);
        let col_count = Self::get_col_count(shape);
        assert!(index < row_count, "Row index out of range {} vs {:?}", index, shape);
        let start = col_count * index;
        let end = col_count * (index + 1);
        data[start..end].to_vec()
    }

    pub fn get_col(index: usize, shape: &Vec<usize>, data: &Vec<T>) -> Vec<T> {
        let row_count = Self::get_row_count(shape);
        let col_count = Self::get_col_count(shape);
        assert!(index < col_count, "Col index out of range {} vs {:?}", index, shape);
        (0..row_count)
            .map(|row| data[col_count * row + index])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::IndexTools;

    fn matrix1234() -> (Vec<f32>, Vec<usize>){
        let data = (1..=4).map(|x| x as f32).collect();
        let shape = vec![2, 2];
        (data, shape)
    }

    #[test]
    fn set_item() {
        let (mut data, shape) = matrix1234();        
        IndexTools::set_item(vec![1, 0], 5.0, &shape, &mut data);
        assert_eq!(data, vec![1.0, 2.0, 5.0, 4.0]);
    }    
    
    #[test]
    fn get_item() {
        let (data, shape) = matrix1234();
        let recieved = IndexTools::get_item(vec![1, 0], &shape, &data);
        assert_eq!(recieved, 3.0);
    }

    #[test]
    fn get_row() {
        let (data, shape) = matrix1234();        
        let recieved = IndexTools::get_row(1, &shape, &data);
        assert_eq!(recieved, vec![3.0, 4.0]);
    }

    #[test]
    fn get_col() {
        let (data, shape) = matrix1234();        
        let recieved = IndexTools::get_col(1, &shape, &data);
        assert_eq!(recieved, vec![2.0, 4.0]);
    }
}