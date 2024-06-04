mod tensor;
mod optimization;

use tensor::Tensor;
use optimization::{systemle::system_le, GradientDescent};

fn f(vector: &Tensor<f64>) -> f64 {
    let w0 = vector.get_v(0);
    let w1 = vector.get_v(1);
    w0.powi(2) + w1.powi(2) + 2.0 * f64::sin(1.5 * (*w0 + *w1)).powi(2) + 2.0
}

fn grad_f(vector: &Tensor<f64>) -> Tensor<f64> {
    let w0 = vector.get_v(0);
    let w1 = vector.get_v(1);
    let common_teil = 3.0 * f64::sin(3.0 * (*w0 + *w1));
    let dw0 = w0 * 2.0 + common_teil;
    let dw1 = w1 * 2.0 + common_teil;
    Tensor::ket(vec![dw0, dw1])
}

fn main() {
    let a = Tensor::matrix(vec![
        vec![1.0, 2.0, 1.0],
        vec![2.0, 1.0, 2.0],
        vec![3.0, 3.0, 1.0]
    ]);
    let b= Tensor::ket(vec![8.0, 10.0, 12.0]);

    let recieved = system_le(&a, &b, 1000, 0.001);

    let expected = Tensor::ket(vec![1.0, 2.0, 3.0]);
    assert_eq!(recieved, expected);
    
    let mut optimizator = GradientDescent::<f64> {
        func: &f,
        grad_func: Some(&grad_f),
        start_point: Tensor::ket(vec![3.0, 3.0]),
        ..Default::default()
    };
    optimizator.run();
    let result = optimizator.result.unwrap();
    println!("minimum: {}", result.value);
    println!("arg: {}", result.arg);
    println!("logs: {:?}", optimizator.logs);
    //println!("history: {:?}", optimizator.log.data);
    
}
