#[macro_export]
macro_rules! assert_scalar {
    ($tensor:expr) => {
        assert!(
            $tensor.is_scalar(),
            "Tensor is not a scalar: shape = {:?}",
            $tensor.shape
        );
    };
}

#[macro_export]
macro_rules! assert_vector {
    ($tensor:expr) => {
        assert!(
            $tensor.is_vector(),
            "Tensor is not a vector: shape = {:?}",
            $tensor.shape
        );
    };
}

#[macro_export]
macro_rules! assert_ket {
    ($tensor:expr) => {
        assert!(
            $tensor.is_ket(),
            "Tensor is not a ket vector: shape = {:?}",
            $tensor.shape
        );
    };
}

#[macro_export]
macro_rules! assert_bra {
    ($tensor:expr) => {
        assert!(
            $tensor.is_bra(),
            "Tensor is not a bra vector: shape = {:?}",
            $tensor.shape
        );
    };
}

#[macro_export]
macro_rules! assert_matrix {
    ($tensor:expr) => {
        assert!(
            $tensor.is_matrix(),
            "Tensor is not a matrix: shape = {:?}",
            $tensor.shape
        );
    };
}

#[macro_export]
macro_rules! assert_square_matrix {
    ($tensor:expr) => {
        assert!(
            $tensor.is_square_matrix(),
            "Tensor is not a square matrix: shape = {:?}",
            $tensor.shape
        );
    };
}

#[macro_export]
macro_rules! assert_shape {
    ($tensor1:ident, $tensor2:ident) => {
        assert!(
            $tensor1.shape == $tensor2.shape,
            "Shapes do not match: {:?} vs {:?}",
            $tensor1.shape,
            $tensor2.shape
        );
    };
}

#[macro_export]
macro_rules! assert_out_of_range {
    ($tensor:ident, $indices:ident) => {
        assert!(
            $tensor.shape.len() == $indices.len()&& 
            $tensor.shape.iter().zip($indices.iter()).any(|(a, b)| a > b),
            "Index out of range: shape = {:?}, indices = {:?}",
            $tensor.shape,
            $indices
        );
    };
}