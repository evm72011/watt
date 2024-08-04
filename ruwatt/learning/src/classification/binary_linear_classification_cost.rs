use num::Float;
use std::{fmt, fmt::{Display, Formatter}};
use super::sigmoid;

#[derive(PartialEq, Clone)]
pub enum BinaryLinearClassificationCost {
    LeastSquaresSigmoid,
    LeastSquaresTanh,
    CrossEntropy,
    Softmax
}

impl Display for BinaryLinearClassificationCost {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            BinaryLinearClassificationCost::LeastSquaresSigmoid => write!(f, "LeastSquaresSigmoid"),
            BinaryLinearClassificationCost::LeastSquaresTanh => write!(f, "LeastSquaresTanh"),
            BinaryLinearClassificationCost::CrossEntropy => write!(f, "CrossEntropy"),
            BinaryLinearClassificationCost::Softmax => write!(f, "Softmax"),
        }
    }
}

impl BinaryLinearClassificationCost {
    pub fn activation<T>(&self, value: T) -> T where T: Float {
        match self {
            BinaryLinearClassificationCost::LeastSquaresSigmoid | 
            BinaryLinearClassificationCost::CrossEntropy => sigmoid(value),
            BinaryLinearClassificationCost::LeastSquaresTanh  | 
            BinaryLinearClassificationCost::Softmax => T::tanh(value)
        }
    }

    pub fn allowed_values<T>(&self) -> Vec<T> where T: Float {
        match self {
            BinaryLinearClassificationCost::LeastSquaresSigmoid | 
            BinaryLinearClassificationCost::CrossEntropy => vec![T::zero(), T::one()],
            BinaryLinearClassificationCost::LeastSquaresTanh  | 
            BinaryLinearClassificationCost::Softmax => vec![-T::one(), T::one()]
        }
    }
}
