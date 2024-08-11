use std::collections::HashMap;
use data_frame::{ApplyClosure, ApplyError, DataFrame, FrameDataCell};
use learning::{confusion_matrix, BinaryLinearClassificationMethod, BinaryLinearClassificationModel};
use statistics::Statistics;
use optimization::GradientDescent;

fn convert_species(value: &FrameDataCell) -> Result<FrameDataCell, ApplyError> {
    if let FrameDataCell::String(value) = value {
        let value = match value.as_str() {
            "setosa" => -1.0,
            "versicolor" => 1.0,
            _ => 2.0
        };
        Ok(FrameDataCell::Number(value))
    } else {
        let msg = String::from("Value in cell is not a string");
        Err(ApplyError(msg))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = DataFrame::<f64>::from_csv("./data/iris.csv", None)?;

    let mut df = df.filter(|row| {
        if let FrameDataCell::String(ref value) = row[4] {
            value != "virginica"
        } else {
            false
        }
    });

    let mut map: HashMap<&str, ApplyClosure::<f64>> = HashMap::new();
    map.insert("species", Box::new(&convert_species));
    df.apply(map)?;
    print!("{}", df);

    let data = df.to_tensor(None);
    let (train_data, test_data) = data.split(0.66, 1);
    let x_train = train_data.get_cols((0..=3).collect())?;  
    let x_train = Statistics::normalize(&x_train);
    let y_train = train_data.col(4)?; 

    let x_test = test_data.get_cols((0..=3).collect())?;  
    let x_test = Statistics::normalize(&x_test);
    let y_test = test_data.col(4)?;

    assert_eq!(x_train.shape, vec![66, 4]);
    assert_eq!(y_train.shape, vec![66, 1]);
    assert_eq!(x_test.shape, vec![34, 4]);
    assert_eq!(y_test.shape, vec![34, 1]);

    let mut model = BinaryLinearClassificationModel {
        method: BinaryLinearClassificationMethod::Softmax,
        optimizator: GradientDescent {
            step_count: 50,
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