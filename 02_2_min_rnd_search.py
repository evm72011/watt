from typing import Callable
import numpy as np
from decorators import check_arr_size, timeit
from common_02 import generate_displacements, displace
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


'''
2.2 Implementing random search in Python
Implement the random search algorithm in Python and repeat the experiment
discussed in Example 2.4.

2.3 Using random search to minimize a nonconvex function

2.4 Random search with diminishing steplength
'''

@timeit
def min_rnd_search(
        func: Callable[[np.ndarray], np.ndarray],
        w0: list,
        step_count: int = 10,
        directions_count=10,
        step_size=0.1,
        decrement_step=False) -> tuple[np.ndarray, float]:
    w = list(map(lambda x: np.array([x]), w0))
    f = func(*w)
    history = np.array([f])
    for step in range(step_count):
        size = step_size / (step + 1) if decrement_step else step_size
        displacements = generate_displacements(len(w), directions_count, size)
        point = np.array(w).flatten()
        points = displace(point, displacements)
        fs = func(*list(points.T))
        pos = np.argmin(fs)
        (w, f) = (points[pos], fs[pos]) if (fs[pos] < f) else (w, f)
        history = np.append(history, f)
    return w, f, history


if __name__ == '__main__':

    def f_2_26(w: np.ndarray) -> np.ndarray:
        return np.sin(3 * w) + 0.3 * w ** 2
    
    @check_arr_size
    def g_2_34(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        return np.tanh(4 * w1 + 4 * w2) + np.maximum(0.4 * w1 ** 2, 1) + 1
    
    @check_arr_size
    def g_2_35(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        return 100*(w2 - w1**2)**2 + (w1 - 1)**2
    

    result = min_rnd_search(f_2_26, [4.5])
    print(result[:2])
    result = min_rnd_search(f_2_26, [-1.5])
    print(result[:2])

    result = min_rnd_search(g_2_34, [2, 2], 8, 1000, 1)
    print(result[:2])
    result = min_rnd_search(g_2_35, [-2, -2], 50, 1000, 1)
    print(result[:2])
    result = min_rnd_search(g_2_35, [-2, -2], 50, 1000, 1, True)
    print(result[:2])
