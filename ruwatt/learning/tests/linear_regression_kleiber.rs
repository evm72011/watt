use std::{fs, error::Error};
use data_frame::{DataFrame, DataFrameReadOptions};
use tensor::Matrix;
use optimization::{GradientDescent, StepSize};
use learning::{ LinearRegression, LinearRegressionCost };
use statistics::estimate_model;

#[test]
fn linear_regression_kleiber() -> Result<(), Box<dyn Error>> {
    let options = DataFrameReadOptions {
        parse_header: false
    };
    let df = DataFrame::<f64>::from_csv("../data/kleibers_law.csv", Some(options))?;

    let mut data = df.to_tensor(None);
    data.apply(|x:f64| x.ln());
    let data = data.tr();
    assert_eq!(data.shape, vec![1498, 2]);

    let (train_data, test_data) = data.split(0.66, 1);
    let x_train = train_data.col(0)?;  
    let y_train = train_data.col(1)?;  
    let x_test = test_data.col(0)?;  
    let y_test = test_data.col(1)?;

    let mut model = LinearRegression {
        cost_function: LinearRegressionCost::LeastSquares,
        optimizator: GradientDescent {
            step_size: StepSize::Newton,
            step_count: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    model.fit(&x_train, &y_train);
    let y_predict = model.predict(&x_test);

    estimate_model(&y_predict, &y_test, 0.4, 0.85)?;

    let train = Matrix::concat_h(x_train, y_train);
    let predict = Matrix::concat_h(x_test, y_predict);

    let folder = "../data/results/kleibers_law/";
    fs::create_dir_all(folder)?;

    let df = DataFrame::from_tensor(&train);
    let file_name = format!("{folder}train.csv");
    df.save_csv(&file_name, true)?;

    let df = DataFrame::from_tensor(&predict);
    let file_name = format!("{folder}test.csv");
    df.save_csv(&file_name, true)?;

    Ok(())
}
