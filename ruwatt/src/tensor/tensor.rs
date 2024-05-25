use num::Float;

pub struct Tensor<T = f32> where T: Float {
    pub shape: Vec<usize>,
    pub data: Vec<T>
}

impl<T> Tensor<T> where T: Float {
    pub fn compare_shape(&self, tensor: &Tensor<T>) {
        let matching = self.shape.iter().zip(&tensor.shape).any(|(&a, &b)| a == b);
        assert!(matching, "Shape mismatch");
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn get(&self, indices: Vec<usize>) -> Option<&T> {
        self.calculate_index(indices).and_then(|idx| self.data.get(idx))
    }

    pub fn set(&mut self, indices: Vec<usize>, value: T) -> Option<()> {
        if let Some(idx) = self.calculate_index(indices) {
            if let Some(elem) = self.data.get_mut(idx) {
                *elem = value;
                return Some(());
            }
        }
        None
    }

    fn calculate_index(&self, indices: Vec<usize>) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut index = 0;
        let mut stride = 1;
        for (i, &dim) in self.shape.iter().rev().enumerate() {
            if indices[self.shape.len() - 1 - i] >= dim {
                return None;
            }
            index += indices[self.shape.len() - 1 - i] * stride;
            stride *= dim;
        }
        Some(index)
    }
}
