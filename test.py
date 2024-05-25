import numpy as np

a = np.array([-1, 2, 3, -4])
print(a)
print(a.shape)

print(a.T)
print(a.T.shape)

b = np.ones((2,3,4))
n = 0
for i in range(2):
  for j in range(3):
    for k in range(4):
      n += 1
      b[i,j,k] = n
print(b.shape)
print(b.T.shape)
print(b)
print(b.T)

#b = np.where(a > 0, 1, -1)
#print(b)
