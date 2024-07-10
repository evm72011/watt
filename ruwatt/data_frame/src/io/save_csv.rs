use std::error::Error;
use std::{io, fs, fs::File, io::Write};
use super::super::DataFrame;

impl DataFrame {

    pub fn save_csv(&self, file_name: &str, header_names: Option<Vec<&str>>) -> Result<(), Box<dyn Error>> {
        let mut file = File::create(file_name)?;
        self.save_header(&mut file, header_names)?;

        //for i in 0..self.row_count() {
        //}
        Ok(())
    }

    fn save_header(&self, file: &mut File, header_names: Option<Vec<&str>>) -> Result<(), Box<dyn Error>> {
        let header_names: Vec<String> = if let Some(header_names) = header_names {
            assert_eq!(header_names.len(), self.headers.len());
            header_names.iter()
                .map(|&name| String::from(name))
                .collect()
        } else {
            self.headers.iter()
                .map(|header| header.name.clone())
                .collect()
        };
        let line = header_names.join(",") + "\n";
        file.write_all(line.as_bytes())?;
        Ok(())
    }
}