use std::{fs, error::Error};
use ruwatt::tensor::{Tensor, Matrix};
use ruwatt::optimization::GradientDescent;
use ruwatt::learning::{ LinearRegression, CostFunction, mse, r2_score };

#[test]
fn linear_regression_auto() -> Result<(), Box<dyn Error>> {
    let mut data = Tensor::<f32>::empty();
    data.read_from_file("./data/auto.csv", Some(vec![8]), Some(vec![0]))?;
    assert_eq!(data.shape, vec![393, 8]);

    let (train_data, test_data) = data.split(0.66);
    println!("foo");
    println!("{:?}", train_data);
    let x_train = train_data.get_cols((1..=7).collect())?;  
    let y_train = train_data.col(0)?;  
    let x_test = test_data.get_cols((1..=7).collect())?;  
    let y_test = test_data.col(0)?;

    let mut model = LinearRegression {
        cost_function: CostFunction::LeastSquares,
        optimizator: GradientDescent {
            step_count: 1000,
            step_size: 3.0,
            ..Default::default()
        },
        ..Default::default()
    };
    model.fit(&x_train, &y_train);
    let y_predict = model.predict(&x_test);

    let estimation = mse(&y_predict, &y_test).to_scalar();
    assert!(estimation < 1.0);
    let estimation = r2_score(&y_predict, &y_test).to_scalar();
    assert!(estimation > 0.85);

    let train = Matrix::concat_h(x_train, y_train);
    let predict = Matrix::concat_h(x_test, y_predict);

    let folder = "./data/results/auto/";
    fs::create_dir_all(folder)?;

    let train_file_name = &format!("{}{}", folder, "train.csv")[..];
    train.save_to_file(train_file_name)?;
    
    let test_file_name = &format!("{}{}", folder, "test.csv")[..];
    predict.save_to_file(test_file_name)?;

    Ok(())
}
