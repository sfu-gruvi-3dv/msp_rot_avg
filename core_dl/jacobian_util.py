"""
Sample code automatically generated on 2019-07-12 19:38:26

by www.matrixcalculus.org

from input

d/dd K*(R-(1/d)*N) = 1/d.^2*K*N

where

K is a matrix
N is a matrix
R is a matrix
d is a scalar

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(K, N, R, d):
    assert isinstance(K, np.ndarray)
    dim = K.shape
    assert len(dim) == 2
    K_rows = dim[0]
    K_cols = dim[1]
    assert isinstance(N, np.ndarray)
    dim = N.shape
    assert len(dim) == 2
    N_rows = dim[0]
    N_cols = dim[1]
    assert isinstance(R, np.ndarray)
    dim = R.shape
    assert len(dim) == 2
    R_rows = dim[0]
    R_cols = dim[1]
    if isinstance(d, np.ndarray):
        dim = d.shape
        assert dim == (1, )
    assert R_cols == N_cols
    assert R_rows == N_rows == K_cols

    functionValue = np.dot(K, (R - ((1 / d) * N)))
    gradient = ((1 / (d ** 2)) * np.dot(K, N))

    return functionValue, gradient

def checkGradient(K, N, R, d):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = float(np.random.randn(1))
    f1, _ = fAndG(K, N, R, d + t * delta)
    f2, _ = fAndG(K, N, R, d - t * delta)
    f, g = fAndG(K, N, R, d)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=0)))

def generateRandomData():
    K = np.random.randn(3, 3)
    N = np.random.randn(3, 3)
    R = np.random.randn(3, 3)
    d = np.random.randn(1)

    return K, N, R, d

if __name__ == '__main__':
    K, N, R, d = generateRandomData()
    functionValue, gradient = fAndG(K, N, R, d)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(K, N, R, d)