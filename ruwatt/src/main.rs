mod tensor;
mod optimization;
mod learning;

use learning::{LinearRegression, CostFunction};
use tensor::Tensor;
use optimization::GradientDescent;

/*
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
    */

fn main() {
    let a = 5; //String::from("foo");
    let b = a;
    println!("{}", a);
    let mut model = LinearRegression {
        cost_function: CostFunction::LeastSquares,
        optimizator: GradientDescent {
            step_count: 1000,
            step_size: 3.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let x_train = Tensor::matrix(vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 4.0],
        vec![0.5, 0.5],
    ]);
    let y_train = Tensor::matrix(vec![
        vec![ 1.0,  2.0,  3.0],    
        vec![12.0, 15.0, 18.0],   
        vec![23.0, 28.0, 33.0],  
        vec![41.0, 49.0, 57.0],  
        vec![ 6.5,  8.5, 10.5],  
    ]);
    model.fit(x_train, y_train);
    println!("coef: {:?}", model.coef);

    let x_test = Tensor::matrix(vec![
        vec![7.0, 5.0],
        vec![5.0, 7.0]
    ]);

    println!("x_test: {:?}", x_test);
    let y_test = model.predict(x_test);
    println!("y_test: {:?}", y_test);

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
