use num::abs;
use ruwatt::tensor::{ Tensor, Vector };
use ruwatt::optimization::GradientDescent;

fn f(vector: &Tensor<f64>) -> f64 {
    let w0 = vector.get_v(0);
    let w1 = vector.get_v(1);
    w0.powi(2) + w1.powi(2) + 2.0 * f64::sin(1.5 * (w0 + w1)).powi(2) + 2.0
}

fn grad_f(vector: &Tensor<f64>) -> Tensor<f64> {
    let w0 = vector.get_v(0);
    let w1 = vector.get_v(1);
    let common_teil = 3.0 * f64::sin(3.0 * (w0 + w1));
    let dw0 = w0 * 2.0 + common_teil;
    let dw1 = w1 * 2.0 + common_teil;
    Vector::ket(vec![dw0, dw1])
}

#[test]
fn gradient_descent_analytic() {
    let mut optimizator = GradientDescent {
        func: &f,
        grad_func: Some(&grad_f),
        start_point: Vector::ket(vec![3.0, 3.0]),
        ..Default::default()
    };
    optimizator.run();
    
    let recieved = optimizator.result.unwrap();
    let expected_arg = Vector::ket(vec![0.0, 0.0]);
    let expected_value = 2.0;
    assert!(recieved.arg.is_near(&expected_arg, 0.001));
    assert!(abs(recieved.value -expected_value) < 0.001);
}

fn f8(vector: &Tensor<f64>) -> f64 {
    (0..8)
        .map(|index| vector.get_v(index).powi(2))
        .sum()
}

#[test]
fn gradient_descent_f8() {
    let mut optimizator = GradientDescent {
        func: &f8,
        start_point: Vector::ket(vec![2.0; 8]),
        ..Default::default()
    };
    optimizator.run();
    
    let recieved = optimizator.result.unwrap();
    let expected_arg = Vector::ket(vec![0.0; 8]);
    let expected_value = 2.0;
    assert!(recieved.arg.is_near(&expected_arg, 0.001));
    assert!(abs(recieved.value -expected_value) < 0.001);
}
