use std::{collections::HashMap, error::Error};

use ruwatt::data_frame::{DataFrame, DataFrameHeader, DataFrameReadOptions, DataType};


#[test]
fn data_frame_io() -> Result<(), Box<dyn Error>> {
    let mut mapper = HashMap::new();
    mapper.insert("crim", |val: String| DataType::Float(val.parse().unwrap()));
    

    let options = DataFrameReadOptions {
        header: Some(DataFrameHeader::Auto),
        mapper: None
    };
    let df = DataFrame::read_csv("./data/boston_housing.csv", Some(options))?;

    df.save_csv("./data/boston_housing_.csv")?;
    Ok(())    
}
