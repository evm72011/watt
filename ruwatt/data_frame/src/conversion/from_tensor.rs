use std::fmt::Debug;
use num::Float;
use tensor::{Tensor, assert_matrix};
use super::super::{DataFrame, FrameDataCell, FrameHeader};

impl<T> DataFrame<T> where T: Float + Debug + Default{
    pub fn from_tensor(tensor: &Tensor<T>) ->  Self {
        assert_matrix!(tensor);

        let data: Vec<Vec<FrameDataCell<T>>> = tensor.rows()
            .map(|item| item.data.iter().map(|&value| FrameDataCell::<T>::Number(value)).collect())
            .collect();
        let mut headers = FrameHeader::<T>::gen_anonym_headers(tensor.col_count());
        headers.iter_mut()
            .for_each(|header| header.data_type = FrameDataCell::Number(Default::default()));
        DataFrame::<T> {
            headers,
            data
        }
    }
}

#[cfg(test)]
mod tests {
    use tensor::Matrix;
    use crate::{mock::df_2x2, DataFrame};

    #[test]
    fn from_tensor() {
        let matrix = Matrix::square(vec![1.0, 2.0, 3.0, 4.0]);
        let recieved = DataFrame::from_tensor(&matrix);
        let expected = df_2x2();
        assert_eq!(recieved, expected)
    }
}
