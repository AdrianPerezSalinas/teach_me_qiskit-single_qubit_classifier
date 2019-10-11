import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import itertools

def circle_labels(X):
    radius = np.sqrt(2 / np.pi)
    Y = np.empty(len(X))
    for i,x in enumerate(X):
        if np.linalg.norm(x) > radius:
            Y[i] = 0
        else:
            Y[i] = 1
        
    return Y


def squares_labels(X):
    Y = np.empty(len(X))
    for i,x in enumerate(X):
        if x[0] > 0 and x[1] > 0: Y[i] = 0
        elif x[0] > 0 and x[1] < 0: Y[i] = 1
        elif x[0] < 0 and x[1] > 0: Y[i] = 2
        else: Y[i] = 3
            
    return Y


def draw_circle(Data, colorbar=False, check=False, title=''):
    X = Data[0]
    Y = Data[1]
    if check:
        cmap = cm.get_cmap('RdYlGn')
        norm = mpl.colors.Normalize(vmin=0,vmax=1)
    else:
        cmap = cm.get_cmap('plasma')
        norm = mpl.colors.Normalize(vmin=0,vmax=1.2)
    
    radius = np.sqrt(2 / np.pi)
    fig, ax = plt.subplots()
    c = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, norm = norm)
    circle1 = plt.Circle([0,0], radius, color='black', fill=False)
    ax.add_artist(circle1)
    ax.set(xlim=[-1,1], ylim=[-1,1],title=title)
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    if colorbar:
        fig.colorbar(c, ax = ax, boundaries=np.arange(0,1.01, 0.01), ticks=[0,1])
    return fig


def colormesh_circle(x, Z):
    radius = np.sqrt(2 / np.pi)
    cmap = cm.get_cmap('plasma')
    norm = mpl.colors.Normalize(vmin=0,vmax=1)
    fig, ax = plt.subplots()
    cf=ax.pcolormesh(x, x, Z.reshape((len(x), len(x))), cmap=cmap, norm=norm, )
    circle1 = plt.Circle([0,0], radius, color='black', fill=False)
    ax.add_artist(circle1)
    ax.set(xlim=[-1,1], ylim=[-1,1])
    ax.set_aspect('equal', 'box')
    fig.colorbar(cf, ax=ax, boundaries=np.arange(0,1.01, 0.01), ticks=[0,1])
    plt.title(r'$|\langle \psi | 0 \rangle|^2$')


def draw_squares(Data, colorbar=False, check=False, title=''):
    X = Data[0]
    Y = Data[1]
    if check:
        cmap = cm.get_cmap('RdYlGn')
        norm = mpl.colors.Normalize(vmin=0,vmax=1)
    else:
        cmap = cm.get_cmap('brg')
        norm = mpl.colors.Normalize(vmin=0,vmax=3)

    fig, ax = plt.subplots()
    c=ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, norm = norm)
    ax.plot([0,0], [-1,1], color='black')
    ax.plot([-1,1], [0,0], color='black')
    ax.set(xlim=[-1,1], ylim=[-1,1], title='')
    ax.set_aspect('equal', 'box')
    if colorbar:
        fig.colorbar(c, ax = ax)
    return fig



def _checks(p, F, Y):
    s=(F > p).astype(int)
    sol = (s == Y).astype(int)
    solution = np.sum(sol)
    return solution


def swipe_check(F, Y):
    P = np.linspace(0, 1, 101)
    s = 0
    p_sol=0
    for p in P:
        solution = _checks(p, F, Y)
        if solution > s:
            s = solution
            p_sol = p
            
    return s, p_sol
