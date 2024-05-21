import numpy as np

a = np.array([-1, 2, 3, -4])
print(a)

b = np.where(a > 0, 1, -1)
print(b)