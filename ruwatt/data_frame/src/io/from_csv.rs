use std::fmt::Debug;
use std::{error::Error, fs::File};
use std::io::{BufRead, BufReader};
use num::Float;
use super::super::{DataFrame, FrameDataCell, FrameHeader};
use super::{DataFrameReadOptions, DataFrameIOError};


impl<T> DataFrame<T> where T: Float + Debug + Default {
    pub fn from_csv(file_name: &str, options: Option<DataFrameReadOptions>) -> Result<DataFrame<T>, Box<dyn Error>> {
        let options = if let Some(options) = options {
            options
        } else {
            DataFrameReadOptions::default()
        };
        let file = File::open(file_name)?;
        let mut reader = BufReader::new(file);
        let mut result = DataFrame::<T>::new();
        result.parse_header(&mut reader, options)?;
        result.parse_body(&mut reader)?;
        Ok(result)
    }

    fn parse_header<R: BufRead>(&mut self, reader: &mut R, options: DataFrameReadOptions) -> Result<(), Box<dyn Error>> {
        if options.parse_header {
            let mut header_line = String::new();
            reader.read_line(&mut header_line)?;

            self.headers = header_line.trim().split(',')
                .map(|value| {
                    let name = value.replace("\"", "").to_string();
                    FrameHeader::new(name)
                })
                .collect();
            Ok(())
        } else {
            Err(Box::new(DataFrameIOError::HeaderParsingError))
        }
    }

    fn parse_body<R: BufRead>(&mut self, reader: &mut R) -> Result<(), Box<dyn Error>> {
        for (line_index, line) in reader.lines().enumerate() {
            match line {
                Ok(line) => {
                    let mut iter = line.trim().split(',').enumerate().peekable();
                    let col_count = iter.clone().count() + 1;
                    let mut line_data = Vec::with_capacity(col_count);
                    while let Some((cell_index, value)) = iter.next() {
                        if self.headers.len() == 0 {
                            self.headers = FrameHeader::<T>::gen_anonym_headers(col_count);
                        }

                        let is_last = iter.peek().is_none();
                        if line_index > 0 {
                            self.validate_line(is_last, cell_index, line_index)?;
                        }
                        let value: FrameDataCell::<T> = value.parse().unwrap();
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

    fn validate_line(&self, is_last: bool, cell_index: usize, line_index: usize) ->Result<(), DataFrameIOError>  {
        if is_last && cell_index + 1 < self.col_count() {
            return Err(DataFrameIOError::NotEnoughDataInLine(line_index));
        }
        if !is_last && cell_index >= self.col_count() {
            return Err(DataFrameIOError::TooMuchDataInLine(line_index));
        }
        Ok(())
    }
}
