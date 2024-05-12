from autograd import grad
from newton import newton
from typing import Callable

'''
3.4 First-order coordinate descent as a local optimization scheme
'''
def get_func(dfs, index, w, x):
    w[index] = x
    return dfs[index](*w)

def min_first_order_descent(
        func: Callable,
        w0: list,
        step_count: int = 10,
) -> tuple[list, float, list]:
    log = []
    w = w0
    dfs = list(map(lambda i: grad(func, i), range(len(w))))
    for step in range(step_count):
        index = step % len(w)
        df = lambda x: get_func(dfs, index, w, x)
        w_index, _ = newton(df, w[index])
        w[index] = w_index
        log.append([*w])
    return w, func(*w), log

if __name__ == '__main__':
    g = lambda w0, w1: w0**2 + w1**2 + 2
    w, f, log = min_first_order_descent(g, [3.0, 4.0], 5)
    print(w, f, log)

    g = lambda w0, w1: 2 * (w0**2 + w1**2 + w0 * w1 + 10)
    w, f, log =  min_first_order_descent(g, [3.0, 4.0], 5)
    print(w, f, log)
