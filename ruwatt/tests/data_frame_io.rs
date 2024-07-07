use std::error::Error;

use ruwatt::data_frame::{DataFrame, DataFrameHeader, DataFrameReadOptions};


#[test]
fn data_frame_io() -> Result<(), Box<dyn Error>> {
    let options = DataFrameReadOptions {
        header: Some(DataFrameHeader::Auto)
    };
    let df = DataFrame::read_csv("./data/boston_housing.csv", Some(options));
    Ok(())    
}
