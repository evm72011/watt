use num::Float;
use tensor::Tensor;

use super::super::DataFrame;

impl<T> DataFrame<T> where T: Float {
    pub fn from_tensor(_tensor: Tensor) ->  Self {
        DataFrame::<T>::new()
    }
}
