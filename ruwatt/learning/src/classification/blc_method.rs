use num::Float;
use std::{fmt, fmt::{Display, Formatter}};
use super::sigmoid;

/// Binary Linear Classification Method
#[derive(PartialEq, Clone)]
pub enum BLCMethod {
    LeastSquaresSigmoid,
    LeastSquaresTanh,
    CrossEntropy,
    Softmax
}

impl Display for BLCMethod {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            BLCMethod::LeastSquaresSigmoid => write!(f, "LeastSquaresSigmoid"),
            BLCMethod::LeastSquaresTanh => write!(f, "LeastSquaresTanh"),
            BLCMethod::CrossEntropy => write!(f, "CrossEntropy"),
            BLCMethod::Softmax => write!(f, "Softmax"),
        }
    }
}

impl BLCMethod {
    pub fn activation<T>(&self, value: T) -> T where T: Float {
        match self {
            BLCMethod::LeastSquaresSigmoid | BLCMethod::CrossEntropy => sigmoid(value),
            BLCMethod::LeastSquaresTanh | BLCMethod::Softmax => T::tanh(value)
        }
    }

    pub fn allowed_values<T>(&self) -> Vec<T> where T: Float {
        match self {
            BLCMethod::LeastSquaresSigmoid | BLCMethod::CrossEntropy => vec![T::zero(), T::one()],
            BLCMethod::LeastSquaresTanh | BLCMethod::Softmax => vec![-T::one(), T::one()]
        }
    }
}
