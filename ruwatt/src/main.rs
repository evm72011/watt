mod tensor;
mod optimization;

use tensor::{dot::dot, Tensor};
use optimization::GradientDescent;

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
    let bra = Tensor::bra(vec![ 1.0, 2.0 ]);
    let ket = Tensor::ket(vec![ 3.0, 4.0 ]);
    let recieved = dot(&bra, &ket);
    println!("recieved: {:?}", recieved);


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
