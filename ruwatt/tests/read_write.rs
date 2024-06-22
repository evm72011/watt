use std::fs;
use std::error::Error;
use std::path::Path;
use ruwatt::tensor::Matrix;

#[test]
fn read_save_to_file() -> Result<(), Box<dyn Error>>{
    let file_name = String::from("./data/matrix.csv");
    let _ = fs::remove_file(&file_name);
    
    let mut matrix = Matrix::new(vec![
        vec![ 1.0, 2.0, 3.0 ], 
        vec![ 4.0, 5.0, 6.0 ],
        vec![ 7.0, 8.0, 9.0 ]
    ]);

    matrix.save_to_file(file_name.clone())?;
    let file_exists = Path::new(&file_name).exists();
    assert!(file_exists);

    matrix.read_from_file(file_name, Some(vec![1]), None)?;
    let expected = Matrix::new(vec![
        vec![ 1.0, 3.0 ], 
        vec![ 4.0, 6.0 ],
        vec![ 7.0, 9.0 ]
    ]);
    assert_eq!(matrix, expected);
    Ok(())
}
