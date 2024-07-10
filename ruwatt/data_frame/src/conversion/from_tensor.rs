use std::fmt::Debug;
use num::Float;
use tensor::Tensor;
use super::super::{DataFrame, FrameDataCell};

impl<T> DataFrame<T> where T: Float + Debug + Default{
    pub fn from_tensor(tensor: Tensor<T>) ->  Self {
        let data: Vec<FrameDataCell<T>> = tensor.data.iter()
            .map(|&value| FrameDataCell::<T>::Number(value))
            .collect();
        let mut result = DataFrame::<T> {
            headers: vec![],
            data
        };
        result.init_anonym_header(tensor.col_count());
        result
    }
}
