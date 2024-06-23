use num::Float;
use std::{io, fs, fs::File, io::Write};
use std::error::Error;
use std::fmt::Display;
use std::str::FromStr;
use super::{ Tensor, IndexTools };
use crate::assert_matrix;

impl<T> Tensor<T> 
where 
    T: Float + Display 
{  
    pub fn save_to_file(&self, file_name: &str) -> Result<(), Box<dyn Error>> {
        assert_matrix!(self);
        let mut file = File::create(file_name)?;
        for i in 0..self.row_count() {
            let row = IndexTools::get_row(i, &self.shape, &self.data)?;
            let row: Vec<String> = row.into_iter().map(|item| item.to_string()).collect();
            let line = row.join(",") + "\n";
            file.write_all(line.as_bytes())?;
        }
        Ok(())
    }
}

impl<T> Tensor<T> 
where 
    T: Float + FromStr, 
    <T as FromStr>::Err: 'static + Error + Send + Sync 
{ 
    pub fn read_from_file(&mut self, file_name: &str, skip_cols: Option<Vec<usize>>, skip_rows: Option<Vec<usize>>) -> Result<(), Box<dyn Error>> {
        let contents = fs::read_to_string(file_name)?;
        self.data = Vec::new();
        let lines: Vec<&str> = contents.lines().collect();
        let row_count = lines.len();
        if row_count == 0 {
            return Err(Box::new(io::Error::new(io::ErrorKind::InvalidData, "No rows found in the file")));
        }

        let skip_cols = skip_cols.unwrap_or_default();
        let col_count = lines[0].split(',').count() - skip_cols.len();
        if col_count == 0 {
            return Err(Box::new(io::Error::new(io::ErrorKind::InvalidData, "No columns found in the file")));
        }

        let skip_rows = skip_rows.unwrap_or_default();
        self.data = lines.iter().enumerate()
            .filter(|(row_idx, _)| !skip_rows.contains(&row_idx))
            .flat_map(|(_, &line)| 
                line.split(',').enumerate()
                    .filter(|(col_idx, _)| !skip_cols.contains(&col_idx))
                    .map(|(_, value)| value.trim().parse::<T>().expect("Parsing error"))
            )
            .collect::<Vec<T>>();
        self.shape = vec![row_count, col_count];
        Ok(())
    }
}