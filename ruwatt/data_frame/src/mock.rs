use super::{DataFrame, FrameHeader, FrameDataCell};

#[allow(dead_code)]
pub fn df_2x2() -> DataFrame::<f64> {
    DataFrame {
        headers: vec![
            FrameHeader {
                name: String::from("foo"),
                data_type: FrameDataCell::Number(Default::default())
            },
            FrameHeader {
                name: String::from("bar"),
                data_type: FrameDataCell::Number(Default::default())
            }
        ],
        data: vec![
            FrameDataCell::numbers(&[1.0, 2.0]),
            FrameDataCell::numbers(&[3.0, 4.0]),
        ]
    }
}

#[allow(dead_code)]
pub fn df_2x3() -> DataFrame {
    DataFrame::<f64> {
        headers: vec![
            FrameHeader {
                name: String::from("foo"),
                data_type: FrameDataCell::Number(Default::default())
            },
            FrameHeader {
                name: String::from("bar"),
                data_type: FrameDataCell::Number(Default::default())
            }
        ],
        data: vec![
            FrameDataCell::numbers(&[1.0, 2.0]),
            FrameDataCell::numbers(&[3.0, 4.0]),
            FrameDataCell::numbers(&[5.0, 6.0]),
        ]
    }
}
