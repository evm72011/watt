from typing import Callable
from autograd import grad
from autograd import numpy as np
from common_02 import normalize


'''
3.9 Code up momentum-accelerated gradient descent
'''


def min_gdma(
        func: Callable,
        w0: list,
        step_count: int = 10,
        step_size=None,
        betta=0.7,
        decrement_step=False
) -> tuple[list, float]:
    w = np.array(w0, dtype=np.float32)
    dfs = list(map(lambda i: grad(func, i), range(len(w))))
    for step in range(step_count):
        grad_f = np.array(list(map(lambda i: dfs[i](*w), range(len(w)))))
        grad_f = grad_f if step == 0 else betta*grad_prev + (1-betta)*grad_f
        grad_prev = grad_f
        if not step_size == None: 
            size = step_size / (step + 1) if decrement_step else step_size
            grad_f = normalize(grad_f, size)
        w = w - grad_f
    return w, func(*w)


if __name__ =='__main__':
    f = lambda w0, w1: 0.5 * w0**2 + 9.75 * w1**2
    print(min_gdma(f, [2, 1], 100, 1, 0.0, True))
    print(min_gdma(f, [2, 1], 100, 1, 0.2, True))
    print(min_gdma(f, [2, 1], 100, 1, 0.7, True))
