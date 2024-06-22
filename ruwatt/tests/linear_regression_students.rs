use std::error::Error;
use ruwatt::tensor::Tensor;
use ruwatt::optimization::GradientDescent;
use ruwatt::learning::{ LinearRegression, CostFunction, mse };

#[test]
fn linear_regression_students_debt() -> Result<(), Box<dyn Error>> {
    let mut y_data = Tensor::<f32>::empty();
    y_data.read_from_file("./data/student_debt.csv".to_string(), Some(vec![0]), None)?;

    let mut data: Tensor<f32> = Tensor {
        shape: y_data.shape.clone(),
        data: (0..y_data.row_count()).map(|x| x as f32).collect()
    };
    data.append_col(y_data);

    let (train_data, test_data) = data.split(0.66);


    let mut x_train = train_data.col(0)?;  
    let y_train = train_data.col(1)?;  
    let mut x_test = test_data.col(0)?;  
    let y_test = test_data.col(1)?;  

    let mut model = LinearRegression {
        cost_function: CostFunction::LeastSquares,
        optimizator: GradientDescent {
            step_count: 1000,
            step_size: 1.0,
            ..Default::default()
        },
        ..Default::default()
    };
    model.fit(&x_train, &y_train);
    let y_predict = model.predict(&x_test);
    let estimation = mse(&y_predict, &y_test).to_scalar();
    assert!(estimation < 1.0);

    x_train.append_col(y_train);
    x_test.append_col(y_predict);
    let folder = "./data/results/student_debt/";
    x_train.save_to_file(format!("{}{}", folder, "train.csv"))?;
    x_test.save_to_file(format!("{}{}", folder, "test.csv"))?;
    Ok(())
}
