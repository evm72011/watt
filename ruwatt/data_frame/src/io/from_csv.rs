use std::fmt::Debug;
use std::{error::Error, fs::File};
use std::io::{BufRead, BufReader, Error as IOError, ErrorKind};
use num::Float;
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
        for (line_index, line) in reader.lines().enumerate() {
            match line {
                Ok(line) => {
                    let mut iter = line.trim().split(',').enumerate().peekable();
                    let col_count = iter.clone().count() + 1;
                    let mut line_data = Vec::with_capacity(col_count);
                    while let Some((cell_index, value)) = iter.next() {
                        if self.col_count() == 0 {
                            self.headers = FrameHeader::<T>::gen_anonym_headers(col_count);
                        }

                        let is_last = iter.peek().is_none();
                        self.validate_line(is_last, cell_index, line_index);

                        let value = FrameDataCell::<T>::from(value);
                        self.set_header_type(cell_index, &value);
                        line_data.push(value);
                    }
                    self.data.push(line_data);
                },
                Err(e) => eprintln!("Error reading line: {}", e)
            }
        }
        Ok(())
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
