use std::error::Error;
use ruwatt::statistics::statistics::Statistics;
use ruwatt::tensor::Tensor;
use ruwatt::optimization::GradientDescent;
use ruwatt::learning::{ LinearRegression, CostFunction, mse, r2_score };

#[test]
#[ignore]
fn linear_regression_auto() -> Result<(), Box<dyn Error>> {
    let mut data = Tensor::<f32>::empty();
    data.read_from_file("./data/auto.csv", Some(vec![8]), Some(vec![0]))?;
    assert_eq!(data.shape, vec![392, 8]);

    data = Statistics::normalize(&data);
    let (train_data, test_data) = data.split(0.66);
    let x_train = train_data.get_cols((1..=7).collect())?;  
    let y_train = train_data.col(0)?;  
    let x_test = test_data.get_cols((1..=7).collect())?;  
    let y_test = test_data.col(0)?;

    let mut model = LinearRegression {
        cost_function: CostFunction::LeastSquares,
        optimizator: GradientDescent {
            ..Default::default()
        },
        ..Default::default()
    };
    model.fit(&x_train, &y_train);
    let y_predict = model.predict(&x_test);

    let estimation = mse(&y_predict, &y_test).to_scalar();
    println!("mse = {:?}", estimation);
    assert!(estimation < 0.25);
    let estimation = r2_score(&y_predict, &y_test).to_scalar();
    println!("r2_score = {:?}", estimation);
    assert!(estimation > 0.75);
    Ok(())
}
