use std::collections::HashMap;
use data_frame::{ApplyChanger, ApplyError, DataFrame, FrameDataCell};
use learning::{confusion_matrix, BinaryLinearClassificationMethod, BinaryLinearClassificationModel};
use statistics::Statistics;
use optimization::GradientDescent;

fn convert_class(value: &FrameDataCell) -> Result<FrameDataCell, ApplyError> {
    if let FrameDataCell::Number(value) = value {
        match value {
            2.0 => Ok(FrameDataCell::Number(-1.0)),
            4.0 => Ok(FrameDataCell::Number(1.0)),
            _ => {
                let msg = format!("The value is out of range: {value}");
                Err(ApplyError(msg))
            }
        }
    } else {
        let msg = String::from("Value in cell is not a number");
        Err(ApplyError(msg))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut df = DataFrame::<f64>::from_csv("./data/breast_cancer_wisconsin.csv", None)?;
    df.drop("id");
    df.remove_na();

    let mut map: HashMap<_, _> = HashMap::new();
    map.insert("class", ApplyChanger::new(Box::new(&convert_class)));
    df.apply(map)?;

    let data = df.to_tensor(None);
    let (train_data, test_data) = data.split(0.8, 1);
    let x_train = train_data.get_cols((0..=8).collect())?;  
    let x_train = Statistics::normalize(&x_train);
    let y_train = train_data.col(9)?; 

    let x_test = test_data.get_cols((0..=8).collect())?;  
    let x_test = Statistics::normalize(&x_test);
    let y_test = test_data.col(9)?;

    assert_eq!(x_train.shape, vec![546, 9]);
    assert_eq!(y_train.shape, vec![546, 1]);
    assert_eq!(x_test.shape, vec![137, 9]);
    assert_eq!(y_test.shape, vec![137, 1]);

    let mut model = BinaryLinearClassificationModel {
        method: BinaryLinearClassificationMethod::Softmax,
        optimizator: GradientDescent {
            step_count: 50,
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    };
    model.fit(&x_train, &y_train);

    let y_predict = model.predict(&x_test);
    let matrix = confusion_matrix(&y_test , &y_predict);
    println!("{}", matrix);

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