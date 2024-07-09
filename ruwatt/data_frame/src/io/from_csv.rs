use std::{error::Error, fs::File};
use std::io::{BufRead, BufReader, Error as IOError, ErrorKind};
use regex::Regex;
use crate::FrameHeader;

use super::super::{DataFrame, FrameData};

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

                self.headers = header_line.trim().split(',')
                    .map(|value| {
                        let name = value.replace("\"", "").to_string();
                        FrameHeader::new(name)
                    })
                    .collect();
                return Ok(());
            }
        }
        Err(Box::new(IOError::new(ErrorKind::InvalidInput, "Header parsing error")))
    }

    fn parse_body<R: BufRead>(&mut self, reader: &mut R) -> Result<(), Box<dyn Error>> {
        let float_pattern = Regex::new(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")?;

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
                            FrameData::String(value[1..value.len()-1].to_string())
                        } else if float_pattern.is_match(value) {
                            FrameData::Number(value.parse().unwrap())
                        } else if value.len() == 0 {
                            FrameData::NA
                        } else {
                            FrameData::String(value.to_string())
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
        self.headers = (0..col_count)
            .map(|i| FrameHeader::new(format!("{i}")))
            .collect();
    }

    fn set_header_type(&mut self, index: usize, value: &FrameData) {
        let value = value.default();
        if FrameData::NA == self.headers[index].data_type {
            self.headers[index].data_type = value;
        } else {
            assert_eq!(self.headers[index].data_type, value);
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