from typing import Callable
import numpy as np
from decorators import check_arr_size
from common_02 import displace, generate_basis


'''
2.8 Coordinate search applied to minimize a simple quadratic
'''


def min_coord_search(
        func: Callable[[np.ndarray], np.ndarray],
        w0: list,
        step_count: int = 10,
        step_size=0.1,
        decrement_step=False) -> tuple[np.ndarray, float]:
    w = list(map(lambda x: np.array([x]), w0))
    f = func(*w)
    history = np.array([f])
    for step in range(step_count):
        size = step_size / (step + 1) if decrement_step else step_size
        displacements = generate_basis(len(w), size)
        point = np.array(w).flatten()
        points = displace(point, displacements)
        fs = func(*list(points.T))
        pos = np.argmin(fs)
        (w, f) = (points[pos], fs[pos]) if (fs[pos] < f) else (w, f)
        history = np.append(history, f)
    return w, f, history


if __name__ == '__main__':
    @check_arr_size
    def g_2_36(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        return w1**2 + w2**2 + 2
    
    @check_arr_size
    def g_2_37(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        return 0.26*(w1**2 + w2**2) - 0.48*w1*w2
    
    print(min_coord_search(g_2_36, [3, 4], 7, 1)[:2])
    
    print(min_coord_search(g_2_37, [2, 2], 50, 1, True)[:2])
    print(min_coord_search(g_2_37, [1, 2], 50, 1, True)[:2])

    print(min_coord_search(g_2_37, [3, 4], 20, 1, True)[:2])

'''
(array([0., 0.]), 2.0)
(array([0.6768566, 0.6572714]), 0.01789487013309965)
(array([0.11867811, 0.11904515]), 0.0005651571880292928)
'''
