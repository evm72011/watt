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

impl DataType {
    pub fn default(&self) -> DataType {
        match self {
            DataType::Bool(_) => DataType::Bool(Default::default()),
            DataType::Float(_) => DataType::Float(Default::default()),
            DataType::String(_) => DataType::String(Default::default()),
            DataType::NA => DataType::NA
        }
    }
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

        for (line_index, line) in reader.lines().enumerate() {
            if line_index == 10 {
                break;
            }
            match line {
                Ok(line) => {
                    let mut iter = line.trim().split(',').enumerate().peekable();
                    while let Some((index, value)) = iter.next() {
                        let is_last = iter.peek().is_none();
                        assert!(!(is_last && index < header_types.len() - 1), 
                                "Not enough data in line {line_index}");
                        let value = if value.starts_with('"') || value.ends_with('"') {
                            DataType::String(value[1..value.len()-1].to_string())
                        } else if bool_pattern.is_match(value) {
                            DataType::Bool(value.parse().unwrap())
                        } else if float_pattern.is_match(value) {
                            DataType::Float(value.parse().unwrap())
                        } else if value.len() == 0 {
                            DataType::NA
                        } else {
                            DataType::String(value.to_string())
                        };
                        Self::set_header_type(&mut header_types, index, &value);
                        result.push(value);
                    }
                },
                Err(e) => eprintln!("Error reading line: {}", e)
            }
        }
        Ok((header_types, result))
    }

    fn set_header_type(header_types: &mut Vec<DataType>, index: usize, value: &DataType) {
        let value = value.default();
        if DataType::NA == header_types[index] {
            header_types[index] = value;
        } else {
            assert_eq!(header_types[index], value);
        }
    }
}

