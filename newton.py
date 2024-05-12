from typing import Callable
from autograd import grad, value_and_grad


def newton(
        f: Callable[[float], float],
        x0: float,
        df: Callable[[float], float] = None,
        delta: float = .001,
        max_iter: int = 100,
        silly = False) -> tuple[float, float]:
    df = grad(f) if df is None else df
    x = x0
    _f = f(x)
    for _ in range(max_iter):
        x = x - f(x) / df(x)
        _f = f(x)
        if abs(_f) < delta: 
            break
    assert abs(_f) < delta or silly
    return x, _f


if __name__ == "__main__":
    f = lambda x: x**2 - 2
    print(newton(f, 1.0))
