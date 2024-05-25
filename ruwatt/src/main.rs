mod tensor;
use num::Float;
use tensor::{Tensor, gradient_descent};

fn test<T>(vector: &Tensor<T>) -> T where T: Float {
    let w0 = vector.get(vec![0]).unwrap();
    let w1 = vector.get(vec![1]).unwrap();
    let f1_5 = T::from(1.5).unwrap();
    let f2_0 = T::from(2.0).unwrap();
    T::powi(*w0, 2) + T::powi(*w1, 2) + f2_0 * T::powi(T::sin(f1_5 * (*w0 + *w1)), 2) + f2_0
}

fn main() {
    let vector = Tensor::<f64>::vector(&[3.0, 3.0]);
    println!("Vector: {}", vector);

    println!("Test: {}", test(&vector));

    let minimum = gradient_descent(&test, vector, 10, 0.1, false);
    println!("minimum: {}", minimum);
}
