use std::{collections::HashMap, error::Error};

use tensor::Tensor;
use data_frame::{DataFrame, DataFrameReadOptions, FrameData};

fn main() -> Result<(), Box<dyn Error>> {
    let tensor: Tensor<f64> = Tensor::random(vec![2, 2]);
    println!("{:?}", tensor);

    let options = DataFrameReadOptions {
        parse_header: true
    };
    let mut df = DataFrame::from_csv("./data/boston_housing_.csv", Some(options))?;
    let row = df.row(0)?;
    let col = df.col(0)?;
    println!("{:?}", row);
    println!("{:?}", col);

    let map = HashMap::new();
    map.insert(String::from("chas"), Box::new(|value: FrameData | value));
    df.apply(map);

    let tensor = df.to_tensor();
    println!("{:?}", tensor);
    df.save_csv("./data/boston_housing_.csv")?;
    Ok(())   
}
