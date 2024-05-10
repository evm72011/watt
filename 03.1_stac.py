from math import sqrt, e
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


task = 3

if task == 1:
    g = lambda w: w * np.log(w) - (1 - w) * np.log(1 - w)
    w1 = (1 + sqrt(1 - 4 / e**2)) / 2
    w2 = (1 - sqrt(1 - 4 / e**2 )) / 2
    x = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    ax.plot(x, g(x))
    ax.scatter([w1, w2], [g(w1), g(w2)], label='tab:blue')

if task == 2:
    g = lambda w: np.log(1 + np.exp(w))
    x = np.linspace(-5, 5, 100)
    fig, ax = plt.subplots()
    ax.plot(x, g(x))

if task == 3:
    g = lambda w: w * np.tanh(w)
    x = np.linspace(-5, 5, 100)
    fig, ax = plt.subplots()
    ax.plot(x, g(x))

plt.show()
