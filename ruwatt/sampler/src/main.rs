use learning::ConfusionMatrix;
use tensor::Vector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Vector::ket(vec![0.0, 1.0, 0.0, 1.0, 0.0 ]);
    let p = Vector::ket(vec![0.0, 1.0, 0.0, 1.0, 1.0 ]);
    let matrix = ConfusionMatrix::new(&a, &p);
    print!("{matrix}");
    Ok(())   
}


/*
struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x: T) -> MyBox<T> {
        MyBox(x)
    }
}
use std::ops::Deref;

impl<T> Deref for MyBox<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> Drop for MyBox<T> {
    fn drop(&mut self) {
        println!("Отбрасывается MyBox");
    }
}

fn hello(name: &str) {
    println!("Здравствуй, {}!", name);
}
*/