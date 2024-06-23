use std::error::Error;
use ruwatt::tensor::Tensor;
use ruwatt::optimization::GradientDescent;
use ruwatt::learning::{ LinearRegression, CostFunction, mse };

#[test]
fn linear_regression_kleiber() -> Result<(), Box<dyn Error>> {
    let mut data = Tensor::<f32>::empty();
    data.read_from_file("./data/kleibers_law.csv", Some(vec![0]), None)?;
    data.apply(|x:f32| x.ln());
    let data = data.tr();
    assert_eq!(data.shape, vec![1497, 2]);

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
    let folder = "./data/results/kleibers_law/";

    let train_file_name = &format!("{}{}", folder, "train.csv")[..];
    x_train.save_to_file(train_file_name)?;
    
    let test_file_name = &format!("{}{}", folder, "test.csv")[..];
    x_test.save_to_file(test_file_name)?;
    Ok(())
}
