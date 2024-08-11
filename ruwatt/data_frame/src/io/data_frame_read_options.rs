pub enum DataValidationBehaviour {
    Panic,
    SetNa,
    SetData
}

pub struct DataFrameReadOptions {
    pub parse_header: bool,
    pub data_validation_behaviour: DataValidationBehaviour
} 

impl Default for DataFrameReadOptions {
    fn default() -> Self {
        Self {
            parse_header: true,
            data_validation_behaviour: DataValidationBehaviour::SetNa
        }
    }
}
