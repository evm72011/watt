use num::Float;
use std::iter::Sum;
use tensor::{assert_ket, dot, Tensor};

use super::sigmoid;

fn log_error<T>(x: &Tensor<T>, w: &Tensor<T>, y: T) -> T where T: Float + Sum {
    let predict = sigmoid(dot(&x.tr(), w).to_scalar());
    let _1 = T::one();
    -y * T::ln(predict) - (_1 - y) * T::ln(_1 - predict)
}

/*Is used for values 0, 1 */
pub fn cros_entropy_cost<T>(x: &Tensor<T>, w: &Tensor<T>, y: &Tensor<T>) -> T where T: Float + Sum {
    assert_ket!(x);
    assert_ket!(w);
    assert_ket!(y);

    let summ = y.data.iter()
        .map(|&value| log_error(x, w, value))
        .sum();
    - ( T::one() / T::from(y.data.len()).unwrap() ) * summ
}

pub fn grad_cros_entropy_cost<T>() -> T where T: Float {
    T::one() // TODO
}

pub fn hessian_cros_entropy_cost<T>() -> T where T: Float {
    T::one() // TODO
}
