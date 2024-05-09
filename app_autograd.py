import autograd.numpy as np
from autograd import grad, value_and_grad
from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


def g(w):
    return np.tanh(w)


def gg(w1, w2):
    return np.tanh(w1 * w2)


if __name__ == '__main__':
    task = 4
    
    if task == 1:
        g_g = grad(g)
        vag_g = value_and_grad(g)
        print(g_g(0.0))
        print(vag_g(0.0))
    
    if task == 2:
        dg = egrad(g)
        dg2 = egrad(dg)
        x = np.linspace(-5, 5, 100)
        fig, ax = plt.subplots()
        ax.plot(x, g(x))
        ax.plot(x, dg(x))
        ax.plot(x, dg2(x))
        
        plt.legend(['g', 'g\'', 'g\'\'']) 
        plt.show()

    if task == 3:
        dg = grad(g)
        dg2 = grad(dg)
        x = np.linspace(-2, 2, 100)
        taylor_1 = lambda x0, x: g(x0) + dg(x0) * (x - x0)
        taylor_2 = lambda x0, x: g(x0) + dg(x0) * (x - x0) + 0.5 * dg2(x0) * (x - x0)**2
        fig, ax = plt.subplots()
        ax.plot(x, g(x))
        ax.plot(x, taylor_1(1.0, x))
        ax.plot(x, taylor_2(1.0, x))
        
        plt.legend(['g', '1st', '2nd']) 
        plt.show()

    if task == 4:
        g1 = egrad(gg, 0)
        g2 = egrad(gg, 1)
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = gg(X, Y)
        Z1 = g1(X, Y)
        Z2 = g2(X, Y)

        fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'})
        axs[0].plot_wireframe(X, Y, Z,  rstride=5, cstride=5)
        axs[1].plot_wireframe(X, Y, Z1, rstride=5, cstride=5)
        axs[2].plot_wireframe(X, Y, Z2, rstride=5, cstride=5)
        plt.show()

