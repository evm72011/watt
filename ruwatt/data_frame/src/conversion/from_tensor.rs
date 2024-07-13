use std::fmt::Debug;
use num::Float;
use tensor::{Tensor, assert_matrix};
use super::super::{DataFrame, FrameDataCell, FrameHeader};

impl<T> DataFrame<T> where T: Float + Debug + Default{
    pub fn from_tensor(tensor: Tensor<T>) ->  Self {
        assert_matrix!(tensor);

        let data: Vec<Vec<FrameDataCell<T>>> = tensor.rows()
            .map(|item| item.data.iter().map(|&value| FrameDataCell::<T>::Number(value)).collect())
            .collect();
        let headers = FrameHeader::<T>::gen_anonym_headers(tensor.col_count());
        DataFrame::<T> {
            headers,
            data
        }
    }
}
