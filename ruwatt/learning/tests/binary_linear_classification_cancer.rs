use std::{collections::HashMap, error::Error};

use data_frame::{ApplyChanger, ApplyError, DataFrame, FrameDataCell};
use learning::{BLCMethod, BLC, ConfusionMatrix};
use optimization::GradientDescent;
use statistics::Statistics;
use tensor::Matrix;

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

#[test]
fn breast_cancer_wisconsin() -> Result<(), Box<dyn Error>> {
    let mut df = DataFrame::<f64>::from_csv("../data/breast_cancer_wisconsin.csv", None)?;
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

    let mut model = BLC {
        method: BLCMethod::Softmax,
        optimizator: GradientDescent {
            step_count: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    model.fit(&x_train, &y_train);

    let y_predict = model.predict(&x_test);
    let recieved = ConfusionMatrix::new(&y_test , &y_predict).to_tensor();

    let expected = Matrix::new(vec![
        vec![97.0,  0.0,    2.0],
        vec![0.0,   0.0,    0.0],
        vec![1.0,   2.0,    35.0]
    ]);
    assert_eq!(recieved, expected);
    Ok(()) 
}