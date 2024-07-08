use std::{error::Error, fs::File};
use std::io::{BufRead, BufReader, Error as IOError, ErrorKind};
use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Bool(bool),
    Float(f64),
    String(String),
    NA
}

pub struct DataFrame {
    pub data: Vec<DataType>,
    pub header_types: Vec<DataType>,
    pub header_names: Vec<String>
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

    pub fn from_csv(file_name: &str, options: Option<DataFrameReadOptions>) -> Result<DataFrame, Box<dyn Error>> {
        let file = File::open(file_name)?;
        let mut reader = BufReader::new(file);
        let header_names = Self::read_header(&mut reader, &options)?;
        let (header_types, data) = Self::read_body(&mut reader, &header_names)?;
        let result = DataFrame {
            header_names,
            header_types,
            data
        };
        Ok(result)
    }

    fn read_header<R: BufRead>(reader: &mut R, options: &Option<DataFrameReadOptions>) -> Result<Vec<String>, Box<dyn Error>> {
        if let Some(opt) = options {
            if opt.parse_header {
                let mut header_line = String::new();
                reader.read_line(&mut header_line)?;
                let result = header_line.trim().split(',')
                    .map(|value| value.replace("\"", "").to_string())
                    .collect();
                return Ok(result);
            }
        }
        Err(Box::new(IOError::new(ErrorKind::InvalidInput, "Header parsing error")))
    }

    fn read_body<R: BufRead>(reader: &mut R, header_names: &Vec<String>) -> Result<(Vec<DataType>,Vec<DataType>), Box<dyn Error>> {
        let float_pattern = Regex::new(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")?;
        let bool_pattern = Regex::new(r"(?i)^\s*(true|false)\s*$")?;
        let mut result = vec![];
        let mut header_types = vec![DataType::NA; header_names.len()];

        for (_index, line) in reader.lines().enumerate() {
            if _index == 10 {
                break;
            }
            match line {
                Ok(line) => {
                    //println!("{}", line);
                    line.trim().split(',').enumerate()
                    .for_each(|(index, value)| {
                        let mut header_type = DataType::NA;
                        if value.starts_with('"') || value.ends_with('"') {
                            result.push(DataType::String(value[1..value.len()-1].to_string()));
                            header_type = DataType::String(Default::default());  
                        } else if bool_pattern.is_match(value) {
                            result.push(DataType::Bool(value.parse().unwrap()));
                            header_type = DataType::Bool(Default::default());
                        } else if float_pattern.is_match(value) {
                            result.push(DataType::Float(value.parse().unwrap()));
                            header_type = DataType::Float(Default::default());
                        } else if value.len() == 0 {
                            result.push(DataType::NA);
                        } else {
                            result.push(DataType::String(value.to_string()));
                            header_type = DataType::String(Default::default());  
                        }
                        Self::set_header_type(&mut header_types, index, header_type);
                    })
                },
                Err(e) => eprintln!("Error reading line: {}", e),
            }
        }
        Ok((header_types, result))
    }

    fn set_header_type(header_types: &mut Vec<DataType>, index: usize, header_type: DataType) {
        if DataType::NA == header_types[index] {
            header_types[index] = header_type;
        } else {
            assert_eq!(header_types[index], header_type);
        }
    }
}

