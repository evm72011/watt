pub mod tensor;
pub mod constructors;
//pub mod operators;
pub mod display;
//pub mod scalar;
//pub mod vector;
//pub mod matrix;
pub mod read_write;
pub mod split;

mod index_tools;

pub use tensor::{ Tensor, TensorType, VectorType };
//pub use operators::*;
//pub use matrix::Matrix;
pub use index_tools::{IndexTools, IndexError};
//pub use vector::Vector;
//pub use scalar::Scalar;
//pub use dot;
