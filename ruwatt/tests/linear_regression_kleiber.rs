use std::{fs, error::Error};
use ruwatt::tensor::{Tensor, Matrix};
use ruwatt::optimization::{GradientDescent, StepSize};
use ruwatt::learning::{ LinearRegression, CostFunction, estimate_model };

#[test]
fn linear_regression_kleiber() -> Result<(), Box<dyn Error>> {
    let mut data = Tensor::<f64>::empty();
    data.read_csv("./data/kleibers_law.csv", Some(vec![0]), None)?;
    data.apply(|x:f64| x.ln());
    let data = data.tr();
    assert_eq!(data.shape, vec![1497, 2]);

    let (train_data, test_data) = data.split(0.66, 1);
    let x_train = train_data.col(0)?;  
    let y_train = train_data.col(1)?;  
    let x_test = test_data.col(0)?;  
    let y_test = test_data.col(1)?;

    let mut model = LinearRegression {
        cost_function: CostFunction::LeastSquares,
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

    let folder = "./data/results/kleibers_law/";
    fs::create_dir_all(folder)?;

    let train_file_name = format!("{folder}train.csv");
    train.save_csv(&train_file_name)?;
    
    let test_file_name = format!("{folder}test.csv");
    predict.save_csv(&test_file_name)?;
    Ok(())
}
