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
  pub columns: Vec<DataColumn>,
}
