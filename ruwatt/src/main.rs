mod tensor;
mod optimization;
mod learning;

use learning::LinearRegression;
use tensor::Tensor;
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
    let mut model = LinearRegression {
        ..Default::default()
    };
    let x_train = Tensor::matrix(vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
    ]);
    println!("x_train: {:?}", x_train.clone());
    let y_train = Tensor::matrix(vec![
        vec![0.0, 1.0, 2.0],    //y_0 = x_0^2 + x_1^2
        vec![2.0, 3.0, 4.0],    //y_1 = x_0^2 + x_1^2 + 1
        vec![8.0, 9.0, 10.0],   //y_2 = x_0^2 + x_1^2 + 2
    ]);
    let x_test = Tensor::ket(vec![3.0, 4.0]);
    model.fit(x_train, y_train);
    let y_test = model.predict(x_test);
    println!("y_test: {:?}", y_test);
    println!("coef: {:?}", model.coef);
    println!("bias: {:?}", model.bias);

    /*    
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
    */
}
