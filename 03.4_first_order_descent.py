from autograd import grad
import autograd.numpy as np
from newton import newton
from typing import Callable

'''
3.4 First-order coordinate descent as a local optimization scheme

'''
def get_func(dfs, pos, w, x):
    w[pos] = x
    return dfs[pos](*w) 

def min_first_order_descent(
        func: Callable,
        w0: list,
        step_count: int = 10,
) -> tuple[np.ndarray, float]:
    log = []
    w = w0
    dfs = list(map(lambda i: grad(func, i), range(len(w))))
    for step in range(step_count):
        pos = step % len(w)
        f = lambda x: get_func(dfs, pos, w, x)
        xi, _ = newton(f, w[pos])
        w[pos] = xi
        log.append([*w])
    return w, func(*w), log

if __name__ == '__main__':
    g = lambda w0, w1: w0**2 + w1**2 + w0 * w1 + 2
    w, f, log = min_first_order_descent(g, [3.0, 4.0], 10)
    print(w, f, log)

    g = lambda w0, w1: 2 * (w0**2 + w1**2 + w0 * w1 + 10)
    w, f, log =  min_first_order_descent(g, [3.0, 4.0], 10)
    print(w, f, log)
