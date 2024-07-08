use std::{fs, error::Error};
use tensor::{Matrix, Tensor};
use optimization::GradientDescent;
use learning::{ LinearRegression, CostFunction };
use statistics::estimate_model;

#[test]
fn linear_regression_students_debt() -> Result<(), Box<dyn Error>> {
    let mut y_data = Tensor::<f64>::empty();
    y_data.read_csv("../data/student_debt.csv", Some(vec![0]), None)?;

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
        cost_function: CostFunction::LeastSquares,
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

    let train_file_name = format!("{}{}", folder, "train.csv");
    train.save_csv(&train_file_name)?;
    
    let test_file_name = format!("{}{}", folder, "test.csv");
    predict.save_csv(&test_file_name)?;
    
    Ok(())
}