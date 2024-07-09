use std::error::Error;
use super::super::DataFrame;

impl DataFrame {

  pub fn save_csv(&self, _file_name: &str) -> Result<(), Box<dyn Error>> {
    //for i in 0..self.row_count() {
    //}
    Ok(())
}
}