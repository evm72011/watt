from typing import Callable
import numpy as np
from decorators import check_arr_size
from common_02 import generate_basis


'''
2.10 Coordinate search versus coordinate descent
'''


def min_coord_desc(
        func: Callable[[np.ndarray], float],
        w0: list,
        step_count: int = 10,
        step_size=0.1,
        decrement_step=False) -> tuple[np.ndarray, float]:
    w = list(map(lambda x: np.array([x]), w0))
    f = func(*w)[0]
    for step in range(step_count):
        size = step_size / (step + 1) if decrement_step else step_size
        displacements = generate_basis(len(w), size)
        np.random.shuffle(displacements)
        w = np.array(w).flatten()
        for dw in displacements:
            w_new = w + dw
            f_new = func(*list(map(lambda x: np.array([x]), w_new)))[0]
            if f_new < f:
                w, f = w_new, f_new
                break
    return w, f


if __name__ == '__main__':
    @check_arr_size
    def g_2_37(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        return 0.26*(w1**2 + w2**2) - 0.48*w1*w2
    print(min_coord_desc(g_2_37, [-1, 1], 50, 1, True))
