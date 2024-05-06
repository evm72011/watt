import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


'''
2.1 Minimizing a quadratic function and the curse of dimensionality
'''
N = 100

def minimize(n: int, p: int): 
  points = np.random.rand(p, n) * 2 - 1
  return np.apply_along_axis(lambda w: np.dot(w, w), axis=1, arr=points).min()

x = np.arange(1, N + 1)
v_min = np.vectorize(minimize)
y1 = v_min(x, 10)
y2 = v_min(x, 11)
y3 = v_min(x, 12)
fig, ax = plt.subplots()
ax.scatter(x, y1, label='tab:blue')
ax.scatter(x, y2, label='tab:orange')
ax.scatter(x, y3, label='tab:green')
plt.show()
