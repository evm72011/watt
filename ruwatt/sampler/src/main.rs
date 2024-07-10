use std::{collections::HashMap, error::Error};
use data_frame::{DataFrame, DataFrameReadOptions, FrameDataCell};

fn convert_chas(value: &FrameDataCell) -> FrameDataCell {
    if let FrameDataCell::String(value) = value {
        let value = if value == "0" { 0.0 } else { 1.0 };
        FrameDataCell::Number(value)
    } else {
        panic!("Value in cell is not a string")
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = DataFrameReadOptions {
        parse_header: true
    };
    let mut df = DataFrame::from_csv("./data/boston_housing_.csv", Some(options))?;

    let mut map: HashMap<&str, Box<dyn Fn(&FrameDataCell) -> FrameDataCell>> = HashMap::new();
    map.insert("chas", Box::new(&convert_chas));

    df.apply(map);
    let row = df.row(0)?;
    println!("{:?}", row);
    println!("------------------------------------------");
    
    let row = df.row(1)?;
    println!("{:?}", row);
    println!("------------------------------------------");

    df.drop("chas");
    let row = df.row(0)?;
    println!("{:?}", row);
    println!("------------------------------------------");

    let tensor = df.to_tensor(None);
    println!("{:?}", tensor);

    let mut map = HashMap::new();
    map.insert("medv", "medved");
    df.rename(map);
    
    df.save_csv("./data/results/boston_housing_2.csv", false)?;
    Ok(())   
}
