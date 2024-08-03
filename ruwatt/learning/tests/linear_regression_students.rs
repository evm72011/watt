use std::{fs, error::Error};
use data_frame::{DataFrame, DataFrameReadOptions};
use tensor::{Matrix, Tensor};
use optimization::GradientDescent;
use learning::{ LinearRegression, LinearRegressionCost };
use statistics::estimate_model;

#[test]
fn linear_regression_students_debt() -> Result<(), Box<dyn Error>> {
    let options = DataFrameReadOptions {
        parse_header: false
    };
    let mut df = DataFrame::<f64>::from_csv("../data/student_debt.csv", Some(options))?;
    df.drop("0");
    let y_data = df.to_tensor(None);

    let mut data: Tensor<f64> = Tensor {
        shape: y_data.shape.clone(),
        data: (0..y_data.row_count()).map(|x| x as f64).collect()
    };
    data.append_col(y_data);

    let (train_data, test_data) = data.split(0.66, 1);
    let x_train = train_data.col(0)?;  
    let y_train = train_data.col(1)?;  
    let x_test = test_data.col(0)?;  
    let y_test = test_data.col(1)?;  

    let mut model = LinearRegression {
        cost_function: LinearRegressionCost::LeastSquares,
        optimizator: GradientDescent {
            ..Default::default()
        },
        ..Default::default()
    };
    model.fit(&x_train, &y_train);
    let y_predict = model.predict(&x_test);

    estimate_model(&y_predict, &y_test, 0.01, 0.95)?;

    let train = Matrix::concat_h(x_train, y_train);
    let predict = Matrix::concat_h(x_test, y_predict);

    let folder = "../data/results/student_debt/";
    fs::create_dir_all(folder)?;

    let df = DataFrame::from_tensor(&train);
    let file_name = format!("{}{}", folder, "train.csv");
    df.save_csv(&file_name, true)?;
    
    let df = DataFrame::from_tensor(&predict);
    let file_name = format!("{}{}", folder, "test.csv");
    df.save_csv(&file_name, true)?;
    
    Ok(())
}
