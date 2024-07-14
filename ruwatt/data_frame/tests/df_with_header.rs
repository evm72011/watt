use std::error::Error;
use data_frame::{df_2x3, DataFrame};

#[test]
fn df_with_header() -> Result<(), Box<dyn Error>> {
    let df = DataFrame::<f64>::from_csv("../data/df_with_header.csv", None)?;
    assert_eq!(df, df_2x3());
    Ok(())
}
