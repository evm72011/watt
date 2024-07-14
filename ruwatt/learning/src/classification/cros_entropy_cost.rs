use num::Float;
use std::iter::Sum;
use tensor::{assert_ket, dot, Tensor};

use super::sigma;

fn log_error<T>(x: &Tensor<T>, w: &Tensor<T>, y: T) -> T where T: Float + Sum {
    let predict = sigma(dot(&x.tr(), w).to_scalar());
    let _1 = T::one();
    -y * T::ln(predict) - (_1 - y) * T::ln(_1 - predict)
}

pub fn cros_entropy_cost<T>(x: &Tensor<T>, w: &Tensor<T>, y: &Tensor<T>) -> T where T: Float + Sum {
    assert_ket!(x);
    assert_ket!(w);
    assert_ket!(y);

    let summ = y.data.iter()
        .map(|&value| log_error(x, w, value))
        .sum();

    - ( T::one() / T::from(y.data.len()).unwrap() ) * summ
}
