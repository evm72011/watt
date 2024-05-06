from typing import Callable
import numpy as np
from common_02 import generate_displacements, displace
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


'''
2.6 Revisiting the curse of dimensionality
'''

def get_descent_ratio(
        func: Callable[[np.ndarray], np.ndarray],
        dim: int,
        directions_count,
        step_size=0.1)-> float:
    point = np.zeros(dim); point[0] = 1
    f = func(point)
    displacements = generate_displacements(dim, directions_count, step_size)
    points = displace(point, displacements)
    fs = np.apply_along_axis(func, axis=1, arr=points)
    return np.sum(fs < f) / fs.size


def f(w: np.ndarray) -> float:
    return np.dot(w, w) + 2

get_descent_ratio_v = np.vectorize(get_descent_ratio)
x = np.arange(25) + 1
y1 = get_descent_ratio_v(f, x, 100); print('1st')
y2 = get_descent_ratio_v(f, x, 1_000); print('2nd')
y3 = get_descent_ratio_v(f, x, 10_000); print('3rd')

fig, ax = plt.subplots()
ax.scatter(x, y1, label='tab:blue')
ax.scatter(x, y2, label='tab:orange')
ax.scatter(x, y3, label='tab:green')
plt.show()