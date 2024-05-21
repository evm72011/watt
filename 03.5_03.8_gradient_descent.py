from typing import Callable
from autograd import grad
from autograd import numpy as np
from common_02 import normalize


'''
3.5 Try out gradient descent
3.6 Compare fixed and diminishing steplength values for a simple example
3.7 Oscillation in the cost function history plot
3.8 Tune fixed steplength for gradient descent
'''

def min_gradient_descent(
        func: Callable,
        w0: list,
        step_count: int = 10,
        step_size=0.1,
        decrement_step=False
) -> tuple[list, float]:
    w = np.array(w0, dtype=np.float32)
    dfs = list(map(lambda i: grad(func, i), range(len(w))))
    for step in range(step_count):
        grad_f = np.array(list(map(lambda i: dfs[i](*w), range(len(w)))))
        size = step_size / (step + 1) if decrement_step else step_size
        grad_f = normalize(grad_f, size)
        w = w - grad_f
    return w, func(*list(w))


if __name__ == '__main__':
    task = 4
    if task == 1:
        g = lambda w: (w**4 + w**2 + 10*w) / 50
        print(min_gradient_descent(g, [2], 1000, 1))
        print(min_gradient_descent(g, [2], 1000, .1))
        print(min_gradient_descent(g, [2], 1000, .01))
        print(min_gradient_descent(g, [2], 1000, 1, True))

    if task == 2:
        g = lambda w: np.abs(w)
        print(min_gradient_descent(g, [2.1], 100, 0.5))
        print(min_gradient_descent(g, [2.1], 100, 0.5, True))

    if task == 3:
        g = lambda w1, w2: w1**2 + w2**2 + 2 * np.sin(1.5 * (w1 + w2))**2 + 2
        print(min_gradient_descent(g, [3, 3], 10, 0.01))
        print(min_gradient_descent(g, [3, 3], 10, 0.1))
        print(min_gradient_descent(g, [3, 3], 10, 1))

    if task == 4:
        def g(*w):
            w_np = np.array(w)
            return np.dot(w_np, w_np)
        w_0 = [10 for _ in range(10)]
        print(min_gradient_descent(g, w_0, 100, 0.001))
        print(min_gradient_descent(g, w_0, 100, 0.1))
        print(min_gradient_descent(g, w_0, 100, 1))
