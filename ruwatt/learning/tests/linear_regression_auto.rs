use std::error::Error;
use statistics::{estimate_model, Statistics};
use tensor::Tensor;
use optimization::{GradientDescent, StepSize};
use learning::LinearRegression;

#[test]
fn linear_regression_auto() -> Result<(), Box<dyn Error>> {
    let mut data = Tensor::<f64>::empty();
    data.read_csv("../data/auto.csv", Some(vec![8]), Some(vec![0]))?;
    assert_eq!(data.shape, vec![392, 8]);

    data = Statistics::normalize(&data);
    let (train_data, test_data) = data.split(0.66, 1);
    let x_train = train_data.get_cols((1..=7).collect())?;  
    let y_train = train_data.col(0)?;  
    let x_test = test_data.get_cols((1..=7).collect())?;  
    let y_test = test_data.col(0)?;

    let mut model = LinearRegression {
        optimizator: GradientDescent {
            step_size: StepSize::Newton,
            step_count: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    model.fit(&x_train, &y_train);
    let y_predict = model.predict(&x_test);

    estimate_model(&y_predict, &y_test, 0.25, 0.8)
}
