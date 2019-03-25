import sys
import putil
import numpy as np
# from pylab import *
import matplotlib.pyplot as plt
from numpy import exp,sqrt
from numpy.linalg import inv

# plot parameters
N    = 100
xmin = -1
xmax = 3.5
ymin = -1
ymax = 3

# GP kernel parameters
eta   = 0.1
tau   = 1
sigma = 1

def kgauss (params):
    [tau,sigma] = params
    return lambda x,y: tau * exp (-(x - y)**2 / (2 * sigma * sigma))

def kv (x, xtrain, kernel):
    return np.array ([kernel(x,xi) for xi in xtrain])

def kernel_matrix (xx, kernel):
    N = len(xx)
    return np.array (
        [kernel (xi, xj) for xi in xx for xj in xx]
    ).reshape(N,N) + eta * np.eye(N)

def gpr (xx, xtrain, ytrain, kernel):
    K = kernel_matrix (xtrain, kernel)
    Kinv = inv(K)
    ypr = []; spr = []
    for x in xx:
        s = kernel (x,x) + eta
        k = kv (x, xtrain, kernel)
        ypr.append (k.T.dot(Kinv).dot(ytrain))
        spr.append (s - k.T.dot(Kinv).dot(k))
    return ypr, spr

def gpplot (xx, xtrain, ytrain, kernel, params):
    ypr,spr = gpr (xx, xtrain, ytrain, kernel(params))
    plt.plot (xtrain, ytrain, 'bx', markersize=16)
    plt.plot (xx, ypr, 'b-')
    plt.fill_between (xx, ypr - 2*sqrt(spr), ypr + 2*sqrt(spr), color='#ccccff')

def usage ():
    print('usage: gpr.py train output')
    print('$Id: gpr.py,v 1.3 2017/11/12 04:51:52 daichi Exp $')
    sys.exit (0)

def main ():
    if len(sys.argv) < 2:
        usage ()
    else:
        train = np.loadtxt (sys.argv[1], dtype=float)
        
    xtrain = train.T[0]
    ytrain = train.T[1]
    kernel = kgauss
    params = [tau,sigma]
    xx     = np.linspace (xmin, xmax, N)

    gpplot (xx, xtrain, ytrain, kernel, params)
    
    # putil.simpleaxis ()
    if len(sys.argv) > 2:
        plt.savefig (sys.argv[2])
    else:
        show ()

if __name__ == "__main__":
    main ()