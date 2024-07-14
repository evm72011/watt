use num::Float;

pub fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + T::exp(-x))
}
