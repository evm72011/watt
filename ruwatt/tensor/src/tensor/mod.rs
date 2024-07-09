pub mod tensor;
pub mod constructors;
pub mod display;
pub mod read_write;
pub mod split;
pub mod index_error;

mod index_tools;

pub use tensor::{ Tensor, TensorType, VectorType };
pub use index_tools::IndexTools;
pub use index_error::IndexError;
