import numbers

from newton import newton
import numpy as np
from typing import Callable
from autograd import grad


def get_func(dfs, pos, w, x):
    w[pos] = x if isinstance(x, numbers.Number) else x._value
    return dfs[pos](w)


def min_first_order_descent(
        func: Callable[[np.ndarray[float]], float],
        w0: list,
        step_count: int = 10,
) -> tuple[np.ndarray, float]:
    w = np.array(w0, dtype='f')
    dfs = list(map(lambda i: grad(func, i), range(w.size)))
    for step in range(step_count):
        pos = step % w.size
        xi, _ = newton(lambda x: get_func(dfs, pos, w, x), w[pos])
        w[pos] = xi
    return w, func(w)


if __name__ == '__main__':
    def f(w: np.ndarray) -> float:
        return 2 * (np.dot(w, w) + w[0] * w[1] + 10)
    #print(min_first_order_descent(f, [3, 4]))

    dfs = list(map(lambda i: grad(f, i), range(2)))
    df0 = lambda x: get_func(dfs, 0, np.ones(2), x)
    w = np.ones(2)
    print(grad(f, 1)(w))
