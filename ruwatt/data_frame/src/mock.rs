use super::{DataFrame, FrameHeader, FrameDataCell};

pub fn df_1234() -> DataFrame {
    DataFrame::<f64> {
        headers: vec![
            FrameHeader {
                name: String::from("0"),
                data_type: FrameDataCell::Number(Default::default())
            },
            FrameHeader {
                name: String::from("1"),
                data_type: FrameDataCell::Number(Default::default())
            }
        ],
        data: vec![
            vec![FrameDataCell::Number(1.0), FrameDataCell::Number(2.0)],
            vec![FrameDataCell::Number(3.0), FrameDataCell::Number(4.0)],
        ]
    }
}
