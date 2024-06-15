use num::Float;
use std::fmt;
use super::Tensor;

fn print_cell<T>(tensor: &Tensor<T>, indices: Vec<usize>, f: &mut fmt::Formatter) -> fmt::Result where T: Float + fmt::Display {
    let value = tensor.get(indices);
    write!(f, "{:>8} ", value)?;
    Ok(())
}

fn print_vector<T>(tensor: &Tensor<T>, f: &mut fmt::Formatter) -> fmt::Result where T: Float + fmt::Display {
    for col in 0..tensor.shape[0] {
        let indices = vec![col];
        print_cell(tensor, indices, f)?;
    }
    writeln!(f)?;
    Ok(())
}

fn print_matrix<T>(tensor: &Tensor<T>, f: &mut fmt::Formatter) -> fmt::Result where T: Float + fmt::Display {
    for row in 0..tensor.shape[0] {
        for col in 0..tensor.shape[1] {
            let indices = vec![row, col];
            print_cell(tensor, indices, f)?;
        }
        writeln!(f)?;
    }
    Ok(())
}

fn print_tensor(f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Print not implemented for dim > 2")?;
    Ok(())
}

impl<T> fmt::Display for Tensor<T> where T: Float + fmt::Display {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      writeln!(f)?;
      if self.is_vector() {
        print_vector(self, f)
      } else if self.is_matrix() {
        print_matrix(self, f)
      } else {
        print_tensor(f)
      }
  }
}
