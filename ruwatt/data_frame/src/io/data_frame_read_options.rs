pub struct DataFrameReadOptions {
    pub parse_header: bool
} 

impl Default for DataFrameReadOptions {
    fn default() -> Self {
        Self {
            parse_header: true
        }
    }
}
