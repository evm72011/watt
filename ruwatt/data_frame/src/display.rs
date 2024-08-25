use num::Float;
use std::fmt::{Formatter, Result, Display};

use super::{DataFrame, FrameDataCell};

fn pad_string(input: &String, length: usize) -> String {
    if input.len() >= length {
        input.to_string()
    } else {
        let padding = " ".repeat(length - input.len());
        format!("{}{}", input, padding)
    }
} 

fn print_header<T>(df: &DataFrame<T>, f: &mut Formatter) -> Result where T: Float {
    let names: Vec<String> = df.headers.iter()
        .map(|header| pad_string(&header.name, 20))
        .collect();
    writeln!(f, "{}", names.join(" "))?;
    Ok(())
}

fn print_row<T>(row: &Vec<FrameDataCell<T>>,  f: &mut Formatter) -> Result where T: Float + Display {
    let cells: Vec<String> = row.iter()
        .map(|cell| match cell {
            FrameDataCell::Number(val) => format!("{}", val),
            FrameDataCell::String(val) => val.clone(),
            FrameDataCell::NA => String::from("NA")
        })
        .map(|value| pad_string(&value, 20))
        .collect();
    writeln!(f, "{}", cells.join(" "))?;
    Ok(())
}

impl<T> Display for DataFrame<T> where T: Float + Display {
    fn fmt(&self, f: &mut Formatter) -> Result {
        print_header(self, f)?;
        if self.row_count() > 10 {
            self.rows().take(5).for_each(|row| print_row(&row, f).unwrap());
            writeln!(f, "...")?;
            let tail: Vec<Vec<FrameDataCell<T>>> = self.rows().rev().take(5).collect();
            tail.iter().rev().for_each(|row| print_row(row, f).unwrap());
        } else {
            self.rows().for_each(|row| print_row(&row, f).unwrap());
        }
        Ok(())
    }
}
