use std::error::Error;
use std::{fs::File, io::Write};
use crate::FrameDataCell;

use super::super::DataFrame;

impl DataFrame {
    pub fn save_csv(&self, file_name: &str, skip_header: bool) -> Result<(), Box<dyn Error>> {
        let mut file = File::create(file_name)?;
        if !skip_header {
            self.save_header(&mut file)?;
        }
        for index in 0..self.row_count() {
            let values: Vec<String> = self.row(index).unwrap().iter()
                .map(|value| match value {
                    FrameDataCell::NA => String::from(""),
                    FrameDataCell::String(value) => value.clone(),
                    FrameDataCell::Number(value) => format!("{value}")
                })
                .collect();
            let line = values.join(",") + "\n";
            file.write_all(line.as_bytes())?;
        }
        Ok(())
    }

    fn save_header(&self, file: &mut File) -> Result<(), Box<dyn Error>> {
        let header_names: Vec<String> = self.headers.iter()
            .map(|header| header.name.clone())
            .collect();
        let line = header_names.join(",") + "\n";
        file.write_all(line.as_bytes())?;
        Ok(())
    }
}
