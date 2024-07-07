use std::{error::Error, fs::File};
use std::io::{BufRead, BufReader, Error as IOError, ErrorKind};
use regex::Regex;

#[derive(Debug)]
pub enum DataType {
    Bool(bool),
    Float(f64),
    String(String),
    NA
}

pub struct DataFrame {
    pub data: Vec<DataType>,
    pub columns: Vec<String>
}

pub struct DataFrameReadOptions {
    pub parse_header: bool
}   


impl DataFrame {
    pub fn save_csv(&self, _file_name: &str) -> Result<(), Box<dyn Error>> {
        //for i in 0..self.row_count() {
        //}
        Ok(())
    }

    pub fn read_csv(file_name: &str, options: Option<DataFrameReadOptions>) -> Result<DataFrame, Box<dyn Error>> {
        let file = File::open(file_name)?;
        let mut reader = BufReader::new(file);
        let columns = Self::read_header(&mut reader, &options)?;
        let data = Self::read_body(&mut reader, true)?;
        let result = DataFrame {
            columns,
            data
        };
        Ok(result)
    }

    fn read_header<R: BufRead>(reader: &mut R, options: &Option<DataFrameReadOptions>) -> Result<Vec<String>, Box<dyn Error>> {
        if let Some(opt) = options {
            if opt.parse_header {
                let mut header_line = String::new();
                reader.read_line(&mut header_line)?;
                let headers = header_line.trim().split(',')
                    .map(|value| value.replace("\"", "").to_string())
                    .collect();
                return Ok(headers);
            }
        }
        Err(Box::new(IOError::new(ErrorKind::InvalidInput, "Header parsing error")))
    }

    fn read_body<R: BufRead>(reader: &mut R, skip_first_line: bool) -> Result<Vec<DataType>, Box<dyn Error>> {
        let float_pattern = Regex::new(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")?;
        let bool_pattern = Regex::new(r"(?i)^\s*(true|false)\s*$")?;
        let mut result = vec![];

        for (index, line) in reader.lines().enumerate() {
            if index == 10 {
                break;
            }
            if skip_first_line && index == 0 {
                //continue;
            }
            match line {
                Ok(line) => {
                    //println!("{}", line);
                    line.trim().split(',')
                    .for_each(|value| {
                        //println!("{value}");
                        if value.starts_with('"') || value.ends_with('"') {
                            result.push(DataType::String(value[1..value.len()-1].to_string()));
                        } else if bool_pattern.is_match(value) {
                            result.push(DataType::Bool(value.parse().unwrap()));
                        } else if float_pattern.is_match(value) {
                            result.push(DataType::Float(value.parse().unwrap()));
                        } else if value.len() == 0 {
                            result.push(DataType::NA);
                        } else {
                            result.push(DataType::String(value.to_string()));
                        }
                    })
                },
                Err(e) => eprintln!("Error reading line: {}", e),
            }
        }
        Ok(result)
    }
}

