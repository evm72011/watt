use std::{collections::HashMap, error::Error};

use data_frame::{ApplyClosure, ApplyError, DataFrame, FrameDataCell};
use learning::{confusion_matrix, BinaryLinearClassification, BinaryLinearClassificationCost};
use optimization::GradientDescent;
use statistics::Statistics;
use tensor::{Matrix, Tensor};

fn get_data(allowed_values: Vec<f64>) -> Result<(Tensor,Tensor,Tensor,Tensor), Box<dyn Error>> {
    let mut df = DataFrame::<f64>::from_csv("../data/iris.csv", None)?;
    let mut map: HashMap<&str, ApplyClosure::<f64>> = HashMap::new();
    map.insert("species", Box::new(&convert_species_to_numbers));
    df.apply(map)?;
    
    let df = df.filter(|row| clear_data(row, &allowed_values));
    let data = df.to_tensor(None);

    let (train_data, test_data) = data.split(0.66, 1);
    let x_train = train_data.get_cols((0..=3).collect())?;  
    let x_train = Statistics::normalize(&x_train);

    let y_train = train_data.col(4)?; 

    let x_test = test_data.get_cols((0..=3).collect())?;  
    let x_test = Statistics::normalize(&x_test);

    let y_test = test_data.col(4)?;
    Ok((x_train, y_train, x_test, y_test))
}

fn clear_data(row: &Vec<FrameDataCell>, allowed_values: &Vec<f64>) ->  bool {
    if let FrameDataCell::Number(val) = row[4] {
        allowed_values.contains(&val)
    } else {
        false
    }
}

fn convert_species_to_numbers(value: &FrameDataCell) -> Result<FrameDataCell, ApplyError> {
    if let FrameDataCell::String(value) = value {
        let value = match value.as_str() {
            "setosa" => -1.0,
            "versicolor" => 0.0,
            "virginica" => 1.0,
            _ => 2.0
        };
        Ok(FrameDataCell::Number(value))
    } else {
        let msg = String::from("Value in cell is not a string");
        Err(ApplyError(msg))
    }
}

#[test]
fn least_squares_sigmoid() -> Result<(), Box<dyn Error>> {
    let allowed_values = vec![0.0, 1.0];
    let (x_train, y_train, x_test, y_test) = get_data(allowed_values)?;

    let mut model = BinaryLinearClassification {
        cost_function: BinaryLinearClassificationCost::LeastSquaresSigmoid,
        optimizator: GradientDescent {
            step_count: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    model.fit(&x_train, &y_train);
    let y_predict = model.predict(&x_test);

    let recieved = confusion_matrix(&y_test , &y_predict);
    let expected = Matrix::new(vec![
        vec![ 17.0, 0.0],
        vec![ 1.0, 16.0],
    ]);
    assert_eq!(recieved, expected);
    Ok(()) 
}

#[test]
fn least_squares_tanh() -> Result<(), Box<dyn Error>> {
    let allowed_values = vec![-1.0, 1.0];
    let (x_train, y_train, x_test, y_test) = get_data(allowed_values)?;

    let mut model = BinaryLinearClassification {
        cost_function: BinaryLinearClassificationCost::LeastSquaresTanh,
        optimizator: GradientDescent {
            step_count: 50,
            ..Default::default()
        },
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

#[test]
fn cross_entropy() -> Result<(), Box<dyn Error>> {
    let allowed_values = vec![0.0, 1.0];
    let (x_train, y_train, x_test, y_test) = get_data(allowed_values)?;

    let mut model = BinaryLinearClassification {
        cost_function: BinaryLinearClassificationCost::CrossEntropy,
        optimizator: GradientDescent {
            step_count: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    model.fit(&x_train, &y_train);
    let y_predict = model.predict(&x_test);

    let recieved = confusion_matrix(&y_test , &y_predict);
    let expected = Matrix::new(vec![
        vec![ 17.0, 0.0],
        vec![ 1.0, 16.0],
    ]);
    assert_eq!(recieved, expected);
    Ok(()) 
}

#[test]
fn softmax() -> Result<(), Box<dyn Error>> {
    let allowed_values = vec![-1.0, 1.0];
    let (x_train, y_train, x_test, y_test) = get_data(allowed_values)?;

    let mut model = BinaryLinearClassification {
        cost_function: BinaryLinearClassificationCost::Softmax,
        optimizator: GradientDescent {
            step_count: 50,
            ..Default::default()
        },
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
