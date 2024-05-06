from typing import Callable
from autograd import grad


def newton(
        f: Callable[[float], float],
        x0: float,
        df: Callable[[float], float] = None,
        delta: float = .001,
        max_iter: int = 100) -> tuple[float, float]:
    df = grad(f) if df is None else df
    x = x0
    for _ in range(max_iter):
        x = x - f(x) / df(x)
        if abs(x) < delta:
            break
    return x, f(x)


if __name__ == "__main__":
    print(newton(lambda x: x**2 - 2, 1.0))
