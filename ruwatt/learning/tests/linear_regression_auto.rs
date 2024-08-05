use std::error::Error;
use statistics::{estimate_model, Statistics};
use optimization::{GradientDescent, StepSize};
use learning::LinearRegressionModel;
use data_frame::DataFrame;

#[test]
fn linear_regression_auto() -> Result<(), Box<dyn Error>> {
    let mut df = DataFrame::<f64>::from_csv("../data/auto.csv", None)?;
    df.drop("name");
    let mut data = df.to_tensor(None);
    assert_eq!(data.shape, vec![392, 8]);

    data = Statistics::normalize(&data);
    let (train_data, test_data) = data.split(0.66, 1);
    let x_train = train_data.get_cols((1..=7).collect())?;  
    let y_train = train_data.col(0)?;  
    let x_test = test_data.get_cols((1..=7).collect())?;  
    let y_test = test_data.col(0)?;

    let mut model = LinearRegressionModel {
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
