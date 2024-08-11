use std::{collections::HashMap, error::Error};
use data_frame::{df_2x3, DataFrame, DataFrameReadOptions, DataValidationBehaviour};

#[test]
fn df_without_header() -> Result<(), Box<dyn Error>> {
    let options = DataFrameReadOptions {
        parse_header: false,
        data_validation_behaviour: DataValidationBehaviour::Panic
    };
    let mut df = DataFrame::<f64>::from_csv("../data/df_without_header.csv", Some(options))?;
    let mut map = HashMap::new();
    map.insert("0", "foo");
    map.insert("1", "bar");
    df.rename(map);

    assert_eq!(df, df_2x3());
    Ok(())
}
