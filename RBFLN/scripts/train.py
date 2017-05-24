"""Train an rbfln model

Usage:
  train.py -m <n> -n <n>
  train.py -h | --help

Options:
  -m <n>        Number of neurons in the hidden layer.
  -n <n>        Number of iterations.
  -h --help     Show this screen.
"""
from docopt import docopt
from RBFLN.rbfln import RBFLN
import numpy as np
import matplotlib.pyplot as plt


def plotit(f, xs):
    ys = [f(x) for x in xs]
    plt.plot(xs, ys)


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Load the data
    Q = 110
    xs = np.array([np.array([float(x)]) for x in range(Q)])
    ts = np.array([float(x**2) for x in range(Q)])

    xs = (xs - np.amin(xs)) / (np.amax(xs) - np.amin(xs))
    ts = (ts - np.amin(ts)) / (np.amax(ts) - np.amin(ts))
    N = 1

    # Read the number of neurons in the hidden layer
    M = int(opts['-m'])

    # Read the number of iterations
    niter = int(opts['-n'])

    model = RBFLN(xs[:101], ts[:101], xs[100:110], ts[100:110], M, N, niter)

    plotit(model.predict, xs)
    plt.plot(xs, ts)
    plt.show()
