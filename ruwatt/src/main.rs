mod tensor;
use num::Float;
use tensor::{Tensor, gradient};

fn test<T>(vector: &Tensor<T>) -> T where T: Float {
    let w0 = vector.get(vec![0]).unwrap();
    let w1 = vector.get(vec![1]).unwrap();
    let w2 = vector.get(vec![2]).unwrap();
    *w0 * *w1 * *w2
}

fn main() {
    //let mut tensor = Tensor::new(vec![2, 3, 4], 0.1);
    //let tensor1 = Tensor::<f32>::identity(2,4);
    //let tensor2 = sin(&tensor1);
    //let tensor3 = &tensor1 + &tensor2;
    let vector = Tensor::<f64>::random(vec![3]);
    //let vector = Tensor::<f64>::ones(vec![3]);

    /*
    println!("Tensor 1: {}", tensor1);
    println!("Tensor 2: {}", tensor2);
    println!("Tensor 3: {}", tensor3);
    */
    println!("Vector: {}", vector);
    println!("Test: {}", test(&vector));
    let grad = gradient(&test, &vector);
    println!("grad: {}", grad);
/*
    println!("Tensor shape: {:?}", tensor.shape());
    println!("Tensor data: {:?}", tensor.data());
    tensor.set(vec![1, 2, 3], 5.0).unwrap();
    println!("Updated tensor data: {:?}", tensor.data());
    println!("Value at [1, 2, 3]: {:?}", tensor.get(vec![1, 2, 3]));

 */
}
