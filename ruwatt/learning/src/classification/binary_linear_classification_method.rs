use num::Float;
use std::{fmt, fmt::{Display, Formatter}};
use super::sigmoid;

#[derive(PartialEq, Clone)]
pub enum BinaryLinearClassificationMethod {
    LeastSquaresSigmoid,
    LeastSquaresTanh,
    CrossEntropy,
    Softmax
}

impl Display for BinaryLinearClassificationMethod {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            BinaryLinearClassificationMethod::LeastSquaresSigmoid => write!(f, "LeastSquaresSigmoid"),
            BinaryLinearClassificationMethod::LeastSquaresTanh => write!(f, "LeastSquaresTanh"),
            BinaryLinearClassificationMethod::CrossEntropy => write!(f, "CrossEntropy"),
            BinaryLinearClassificationMethod::Softmax => write!(f, "Softmax"),
        }
    }
}

impl BinaryLinearClassificationMethod {
    pub fn activation<T>(&self, value: T) -> T where T: Float {
        match self {
            BinaryLinearClassificationMethod::LeastSquaresSigmoid | 
            BinaryLinearClassificationMethod::CrossEntropy => sigmoid(value),
            BinaryLinearClassificationMethod::LeastSquaresTanh  | 
            BinaryLinearClassificationMethod::Softmax => T::tanh(value)
        }
    }

    pub fn allowed_values<T>(&self) -> Vec<T> where T: Float {
        match self {
            BinaryLinearClassificationMethod::LeastSquaresSigmoid | 
            BinaryLinearClassificationMethod::CrossEntropy => vec![T::zero(), T::one()],
            BinaryLinearClassificationMethod::LeastSquaresTanh  | 
            BinaryLinearClassificationMethod::Softmax => vec![-T::one(), T::one()]
        }
    }
}
