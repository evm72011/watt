use std::error::Error;

use tensor::Tensor;
use data_frame::{DataFrame, DataFrameReadOptions};

fn main() -> Result<(), Box<dyn Error>> {
    let tensor: Tensor<f64> = Tensor::random(vec![2, 2]);
    println!("{:?}", tensor);

    let options = DataFrameReadOptions {
        parse_header: true
    };
    let df = DataFrame::from_csv("./data/boston_housing_.csv", Some(options))?;
    println!("{:?}", df.header_names);
    println!("-----------------------------------------------");
    println!("{:?}", df.header_types);
    println!("-----------------------------------------------");
    println!("{:?}", df.data);

    df.save_csv("./data/boston_housing_.csv")?;
    Ok(())   
}
