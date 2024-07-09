use std::{error::Error, fs::File};
use std::io::{BufRead, BufReader, Error as IOError, ErrorKind};
use regex::Regex;
use super::super::{DataFrame, DataType};

pub struct DataFrameReadOptions {
    pub parse_header: bool
} 

impl DataFrame {
    pub fn from_csv(file_name: &str, options: Option<DataFrameReadOptions>) -> Result<DataFrame, Box<dyn Error>> {
        let file = File::open(file_name)?;
        let mut reader = BufReader::new(file);
        let mut result = DataFrame::new();
        result.parse_header(&mut reader, &options)?;
        result.parse_body(&mut reader)?;
        Ok(result)
    }

    fn parse_header<R: BufRead>(&mut self, reader: &mut R, options: &Option<DataFrameReadOptions>) -> Result<(), Box<dyn Error>> {
        if let Some(opt) = options {
            if opt.parse_header {
                let mut header_line = String::new();
                reader.read_line(&mut header_line)?;
                self.header_names = header_line.trim().split(',')
                    .map(|value| value.replace("\"", "").to_string())
                    .collect();
                self.header_types = vec![DataType::NA; self.header_names.len()];
                return Ok(());
            }
        }
        Err(Box::new(IOError::new(ErrorKind::InvalidInput, "Header parsing error")))
    }

    fn parse_body<R: BufRead>(&mut self, reader: &mut R) -> Result<(), Box<dyn Error>> {
        let float_pattern = Regex::new(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")?;
        let bool_pattern = Regex::new(r"(?i)^\s*(true|false)\s*$")?;

        for (line_index, line) in reader.lines().enumerate() {
            match line {
                Ok(line) => {
                    let mut iter = line.trim().split(',').enumerate().peekable();
                    while let Some((cell_index, value)) = iter.next() {
                        if self.col_count() == 0 {
                            let col_count = iter.clone().count();
                            self.init_anonym_header(col_count);
                        }

                        let is_last = iter.peek().is_none();
                        self.validate_line(is_last, cell_index, line_index);

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
                        self.set_header_type(cell_index, &value);
                        self.data.push(value);
                    }
                },
                Err(e) => eprintln!("Error reading line: {}", e)
            }
        }
        Ok(())
    }

    fn init_anonym_header(&mut self, col_count: usize) {
        self.header_types = vec![DataType::NA; col_count];
        self.header_names = (0..col_count).map(|i| format!("{i}")).collect();
    }

    fn set_header_type(&mut self, index: usize, value: &DataType) {
        let value = value.default();
        if DataType::NA == self.header_types[index] {
            self.header_types[index] = value;
        } else {
            assert_eq!(self.header_types[index], value);
        }
    }

    fn validate_line(&self, is_last: bool, cell_index: usize, line_index: usize) {
        if is_last && cell_index < self.col_count() - 1 {
            panic!("Not enough data in line {line_index}");
        }

        if !is_last && cell_index >= self.col_count() - 1 {
            panic!("Too much data in line {line_index}");
        }
    }
}