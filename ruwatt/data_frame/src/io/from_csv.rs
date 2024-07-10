use std::fmt::Debug;
use std::{error::Error, fs::File};
use std::io::{BufRead, BufReader, Error as IOError, ErrorKind};
use num::Float;
use regex::Regex;
use super::super::{DataFrame, FrameDataCell, FrameHeader};

pub struct DataFrameReadOptions {
    pub parse_header: bool
} 

impl<T> DataFrame<T> where T: Float + Debug + Default {
    pub fn from_csv(file_name: &str, options: Option<DataFrameReadOptions>) -> Result<DataFrame<T>, Box<dyn Error>> {
        let file = File::open(file_name)?;
        let mut reader = BufReader::new(file);
        let mut result = DataFrame::<T>::new();
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
            }
            return Ok(());
        }
        Err(Box::new(IOError::new(ErrorKind::InvalidInput, "Header parsing error")))
    }

    fn parse_body<R: BufRead>(&mut self, reader: &mut R) -> Result<(), Box<dyn Error>> {
        let number_pattern = Regex::new(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")?;

        for (line_index, line) in reader.lines().enumerate() {
            match line {
                Ok(line) => {
                    let mut iter = line.trim().split(',').enumerate().peekable();
                    while let Some((cell_index, value)) = iter.next() {
                        if self.col_count() == 0 {
                            let col_count = iter.clone().count() + 1;
                            self.init_anonym_header(col_count);
                        }

                        let is_last = iter.peek().is_none();
                        self.validate_line(is_last, cell_index, line_index);

                        let value = if value.starts_with('"') || value.ends_with('"') {
                            FrameDataCell::<T>::String(value[1..value.len()-1].to_string())
                        } else if number_pattern.is_match(value) {
                            let value: f64 = value.parse().unwrap();
                            FrameDataCell::<T>::Number(T::from(value).unwrap())
                        } else if value.len() == 0 {
                            FrameDataCell::<T>::NA
                        } else {
                            FrameDataCell::String(value.to_string())
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

    fn validate_line(&self, is_last: bool, cell_index: usize, line_index: usize) {
        if is_last && cell_index < self.col_count() - 1 {
            panic!("Not enough data in line {line_index}");
        }

        if !is_last && cell_index > self.col_count() - 1 {
            panic!("Too much data in line {line_index}");
        }
    }
}