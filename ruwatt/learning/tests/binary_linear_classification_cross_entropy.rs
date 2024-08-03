use std::{collections::HashMap, error::Error};

use data_frame::{ApplyClosure, ApplyError, DataFrame, FrameDataCell};
use learning::{confusion_matrix, BinaryLinearClassification, BinaryLinearClassificationCost};
use statistics::Statistics;
use tensor::Matrix;

#[test]
fn linear_regression_auto() -> Result<(), Box<dyn Error>> {
    let df = DataFrame::<f64>::from_csv("../data/iris.csv", None)?;
    let mut df = df.filter(|row| remove_values(row, "virginica"));
    let mut map: HashMap<&str, ApplyClosure::<f64>> = HashMap::new();
    map.insert("species", Box::new(&convert_species));
    df.apply(map)?;

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
    
    let mut model = BinaryLinearClassification {
        cost_function: BinaryLinearClassificationCost::CrossEntropy,
        ..Default::default()
    };
    model.fit(&x_train, &y_train);
    let y_predict = model.predict(&x_test);

    let recieved = confusion_matrix(&y_test , &y_predict);
    let expected = Matrix::new(vec![
        vec![ 18.0, 0.0],
        vec![ 0.0, 16.0],
    ]);
    assert_eq!(recieved, expected);
    Ok(())  
}

fn convert_species(value: &FrameDataCell) -> Result<FrameDataCell, ApplyError> {
    if let FrameDataCell::String(value) = value {
        let value = match value.as_str() {
            "setosa" => 0.0,
            "versicolor" => 1.0,
            _ => 2.0
        };
        Ok(FrameDataCell::Number(value))
    } else {
        let msg = String::from("Value in cell is not a string");
        Err(ApplyError(msg))
    }
}

fn remove_values(row: &Vec<FrameDataCell>, value: &str) ->  bool {
    if let FrameDataCell::String(ref val) = row[4] {
        val != value
    } else {
        false
    }
}   