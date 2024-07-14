use num::Float;

pub fn sigma<T: Float>(x: T) -> T {
    T::one() / (T::one() + T::exp(-x))
}
