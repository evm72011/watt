use std::error::Error;

use ruwatt::data_frame::{DataFrame, DataFrameReadOptions};


#[test]
fn data_frame_io() -> Result<(), Box<dyn Error>> {

    let options = DataFrameReadOptions {
        parse_header: true
    };
    let df = DataFrame::read_csv("./data/boston_housing.csv", Some(options))?;
    println!("{:?}", df.columns);
    println!("{:?}", df.data);

    df.save_csv("./data/boston_housing_.csv")?;
    Ok(())    
}
