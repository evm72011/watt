use std::{error::Error, fs::File, io::{BufRead, BufReader}};

pub enum DataType {
    Bool(bool),
    Number(f32),
    String(String)
}

pub struct DataColumn {
    pub data_type: DataType,
    pub name: String
}

pub struct DataFrame {
    pub data: Vec<Option<DataType>>,
    pub columns: Vec<DataColumn>
}

pub enum DataFrameHeaderOption {
    Fixed(Vec<String>),
    Auto
}

pub struct DataFrameReadOptions {
    header: Option<DataFrameHeaderOption>
}   

type Foo = Result<(), Box<dyn Error>>;

impl DataFrame {
    pub fn save_csv(&self, file_name: &str) -> Foo {
        //for i in 0..self.row_count() {

        //}
        Ok(())
    }

    pub fn read_csv(file_name: &str, options: Option<DataFrameReadOptions>) -> Foo {
        let file = File::open(file_name)?;
        let reader = BufReader::new(file);
        let headers = Self::read_header(&mut reader, &options)?;
        Self::read_body(&mut reader, headers)?;
    }

    fn read_header<R: BufRead>(reader: &mut R, options: &Option<DataFrameReadOptions>) -> io::Result<Option<Vec<String>>> {
        if let Some(opt) = options {
            if let Some(hh) = &opt.header {
                match hh {
                    DataFrameHeaderOption::Fixed(headers) => {
                        println!("Headers: {:?}", headers);
                        return Ok(Some(headers.clone()));
                    }
                    DataFrameHeaderOption::Auto => {
                        let mut header_line = String::new();
                        reader.read_line(&mut header_line)?;
                        let headers: Vec<String> = header_line.trim().split(',').map(|s| s.to_string()).collect();
                        println!("Auto headers: {:?}", headers);
                        return Ok(Some(headers));
                    }
                }
            }
        }
        Ok(None)
    }

    fn read_body<R: BufRead>(reader: &mut R, headers: Option<Vec<String>>) -> io::Result<()> {
        for line in reader.lines() {
            match line {
                Ok(line) => println!("{}", line),
                Err(e) => eprintln!("Error reading line: {}", e),
            }
        }
        Ok(())
    }
}